/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.getCuVSResourcesInstance;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraQuery;
import com.nvidia.cuvs.CagraSearchParams;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.FilterBitsetHandle;
import com.nvidia.cuvs.MultiPartitionCagraSearch;
import com.nvidia.cuvs.MultiPartitionSearchResults;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;

/**
 * Extends {@link KnnFloatVectorQuery} for GPU-only search.
 *
 * <p>When all index segments use {@link CuVS2510GPUVectorsReader}, {@link #rewrite} delegates a
 * single multi-partition search to cuVS, passing one Lucene segment per cuVS partition. cuVS
 * runs the per-partition CAGRA searches, applies distance post-processing, and performs the
 * cross-partition top-k merge internally; the returned arrays are mapped to Lucene doc IDs on
 * the host. The effective CAGRA algorithm (SINGLE_CTA or MULTI_KERNEL) is selected by cuVS
 * based on {@code searchAlgo} and {@code itopk_size}, with MULTI_KERNEL handling k beyond
 * SINGLE_CTA's per-partition cap.
 *
 * <p>If the query has an explicit {@code filter}, or if any segment carries live-document deletes,
 * the acceptance mask (filter ∩ liveDocs) is packed into one {@link FilterBitsetHandle} per segment
 * and passed as that partition's filter. The host-side packed arrays are cached per unique
 * (filter, single-segment reader key, field) triple via {@link FilterBitsetCache}; the device
 * upload is cached inside the handle itself across threads. An explicit filter is still evaluated
 * on every query — its per-segment cardinality is what decides whether an approximate search may
 * run at all — so the cache saves the ordinal packing and the upload, not the filter evaluation.
 *
 * <p>Falls back to the standard per-segment Lucene path when the optimized path cannot be
 * applied: mixed segment types, a missing CAGRA index for the field on any segment, segments
 * whose built CAGRA graphs differ in degree (a single multi-partition request requires a uniform
 * graph degree, and a small segment can have its degree truncated at build time), or a filter
 * under which {@link KnnFloatVectorQuery}'s documented exact-search strategy applies (see
 * {@link #rewrite}).
 *
 * @since 25.10
 */
public class GPUKnnFloatVectorQuery extends KnnFloatVectorQuery {

  private final int iTopK;
  private final int searchWidth;
  private final int threadBlockSize;
  private final int maxIterations;
  private final CagraSearchParams.SearchAlgo searchAlgo;

  /**
   * Initializes {@link GPUKnnFloatVectorQuery} with {@link CagraSearchParams.SearchAlgo#AUTO},
   * and max_iterations auto-selected (0).
   *
   * @param field       the vector field name
   * @param target      the query vector
   * @param k           the number of nearest neighbors to return
   * @param filter      optional pre-filter query
   * @param iTopK       CAGRA itopk_size parameter
   * @param searchWidth CAGRA search_width parameter
   */
  public GPUKnnFloatVectorQuery(
      String field, float[] target, int k, Query filter, int iTopK, int searchWidth) {
    this(field, target, k, filter, iTopK, searchWidth, 0, 0, CagraSearchParams.SearchAlgo.AUTO);
  }

  /**
   * Initializes {@link GPUKnnFloatVectorQuery}.
   *
   * @param field           the vector field name
   * @param target          the query vector
   * @param k               the number of nearest neighbors to return
   * @param filter          optional pre-filter query
   * @param iTopK           CAGRA itopk_size parameter
   * @param searchWidth     CAGRA search_width parameter
   * @param threadBlockSize CAGRA thread_block_size (0 = auto)
   * @param maxIterations   CAGRA max_iterations (0 = auto)
   * @param searchAlgo      CAGRA search algorithm
   */
  public GPUKnnFloatVectorQuery(
      String field,
      float[] target,
      int k,
      Query filter,
      int iTopK,
      int searchWidth,
      int threadBlockSize,
      int maxIterations,
      CagraSearchParams.SearchAlgo searchAlgo) {
    super(field, target, k, filter);
    this.iTopK = iTopK;
    this.searchWidth = searchWidth;
    this.threadBlockSize = threadBlockSize;
    this.maxIterations = maxIterations;
    this.searchAlgo = searchAlgo;
  }

  // -------------------------------------------------------------------------
  // Optimized multi-segment path
  // -------------------------------------------------------------------------

  @Override
  public Query rewrite(IndexSearcher indexSearcher) throws IOException {
    IndexReader reader = indexSearcher.getIndexReader();
    List<LeafReaderContext> leaves = reader.leaves();
    if (leaves.isEmpty()) {
      return new MatchNoDocsQuery();
    }

    // Collect a CuVS2510GPUVectorsReader for every segment; fall back if any segment lacks one,
    // has no CAGRA index for this field, or has a CAGRA graph whose degree differs from the other
    // segments'. The multi-partition search requires every partition to share one graph degree
    // (a small segment can have its degree truncated at build time), so a non-uniform set is
    // routed to the per-segment path instead.
    List<CuVS2510GPUVectorsReader> gpuReaders = new ArrayList<>(leaves.size());
    long commonGraphDegree = -1;
    for (LeafReaderContext ctx : leaves) {
      CuVS2510GPUVectorsReader gpuReader = unwrapGpuReader(ctx, field);
      if (gpuReader == null) {
        return super.rewrite(indexSearcher);
      }
      CagraIndex cagraIndex = gpuReader.getCagraIndexForField(field);
      if (cagraIndex == null) {
        return super.rewrite(indexSearcher);
      }
      long graphDegree = cagraIndex.getGraphDegree();
      if (commonGraphDegree == -1) {
        commonGraphDegree = graphDegree;
      } else if (graphDegree != commonGraphDegree) {
        return super.rewrite(indexSearcher);
      }
      gpuReaders.add(gpuReader);
    }

    // Build one filter handle per segment encoding (filter ∩ that segment's liveDocs) whenever any
    // filtering is required — either an explicit Lucene filter, or live-document deletes in at
    // least
    // one segment. Each segment's handle becomes that partition's filter; a segment with neither an
    // explicit filter nor deletes gets a null entry (unfiltered for that partition).
    boolean hasExplicitFilter = (filter != null);
    boolean hasDeletes = false;
    for (LeafReaderContext ctx : leaves) {
      if (ctx.reader().getLiveDocs() != null) {
        hasDeletes = true;
        break;
      }
    }

    // Evaluate an explicit filter once per segment, up front. The resulting bit sets serve twice:
    // they are the acceptance masks the partitions are searched with, and their cardinality is the
    // filter cost that decides whether an approximate search may run at all. KnnFloatVectorQuery
    // documents that a filter leaving at most k candidates in a segment is answered by an exact
    // scan, so those documents come back whatever their distance; CAGRA is approximate and can
    // miss them. A query that selective is therefore handed to the per-segment Lucene path, which
    // applies that rule itself. The whole query goes, not just the selective segment: one
    // multi-partition request cannot leave a single partition out of the GPU search.
    FixedBitSet[] segmentAcceptDocs = null;
    if (hasExplicitFilter) {
      Weight filterWeight = createFilterWeight(indexSearcher);
      segmentAcceptDocs = new FixedBitSet[leaves.size()];
      for (int i = 0; i < leaves.size(); i++) {
        segmentAcceptDocs[i] = evalFilter(filterWeight, leaves.get(i));
        if (segmentAcceptDocs[i].cardinality() <= k) {
          return super.rewrite(indexSearcher);
        }
      }
    }

    // Build the per-segment CagraIndex list (one entry per Lucene segment / cuVS partition).
    CuVSResources resources = getCuVSResourcesInstance();
    List<CagraIndex> cagraIndices = new ArrayList<>(leaves.size());

    List<FilterBitsetHandle> filterHandles = null;
    try {
      if (hasExplicitFilter || hasDeletes) {
        filterHandles = buildPerSegmentFilterHandles(leaves, gpuReaders, segmentAcceptDocs);
      }

      float[] target = getTargetCopy();
      CagraSearchParams searchParams =
          new CagraSearchParams.Builder()
              .withItopkSize(Math.max(iTopK, k))
              .withSearchWidth(searchWidth)
              .withThreadBlockSize(threadBlockSize)
              .withMaxIterations(maxIterations)
              .withAlgo(searchAlgo)
              .build();

      // Upload the query vector to device once; the same matrix view is searched against every
      // partition by the cuVS multi-partition API.
      CuVSMatrix.Builder<?> vectorBuilder =
          CuVSMatrix.deviceBuilder(resources, 1, target.length, CuVSMatrix.DataType.FLOAT);
      vectorBuilder.addVector(target);

      ScoreDoc[] scoreDocs;
      try (CuVSMatrix queryVector = vectorBuilder.build()) {
        for (int i = 0; i < leaves.size(); i++) {
          cagraIndices.add(gpuReaders.get(i).getCagraIndexForField(field));
        }

        CagraQuery cagraQuery =
            new CagraQuery.Builder(resources)
                .withTopK(k)
                .withSearchParams(searchParams)
                .withQueryVectors(queryVector)
                .build();

        MultiPartitionSearchResults results =
            MultiPartitionCagraSearch.search(resources, cagraIndices, cagraQuery, k, filterHandles);

        // The last of the three strategies KnnFloatVectorQuery documents: an approximate search
        // that could not complete under a filter is redone exactly. Every segment cleared the
        // cost > k check above, so more than k candidates were accepted and a search that found
        // them all would have returned k; fewer means CAGRA under-filled its top-k (the unfilled
        // slots carry a sentinel distance and are already dropped from count()). The per-segment
        // Lucene path rescues that case per segment, so hand the query over.
        if (hasExplicitFilter && results.count() < k) {
          return super.rewrite(indexSearcher);
        }

        if (results.count() == 0) {
          return new MatchNoDocsQuery();
        }

        // Map (segmentIdx, ordinal) → global Lucene doc ID; compute normalized score.
        // Stash FloatVectorValues per segment: results.count() can be far larger than the
        // number of segments, and getFloatVectorValues() allocates on each call.
        FloatVectorValues[] segVectorValues = new FloatVectorValues[leaves.size()];
        scoreDocs = new ScoreDoc[results.count()];
        for (int j = 0; j < results.count(); j++) {
          int segIdx = results.getPartitionIndex(j);
          int ordinal = results.getOrdinal(j);
          float dist = results.getDistance(j);

          FloatVectorValues fvv = segVectorValues[segIdx];
          if (fvv == null) {
            fvv = gpuReaders.get(segIdx).getFloatVectorValues(field);
            segVectorValues[segIdx] = fvv;
          }
          LeafReaderContext ctx = leaves.get(segIdx);
          int globalDoc = ctx.docBase + fvv.ordToDoc(ordinal);
          float score = 1.0f / (1.0f + dist);
          scoreDocs[j] = new ScoreDoc(globalDoc, score);
        }

        Arrays.sort(scoreDocs, Comparator.comparingDouble((ScoreDoc sd) -> sd.score).reversed());

        return docAndScoreQuery(scoreDocs);
      }

    } catch (Throwable t) {
      Utils.handleThrowable(t);
      throw new AssertionError("handleThrowable always throws"); // unreachable
    } finally {
      // Release this query's reference on each per-segment handle. A cached handle is kept alive by
      // the cache's own reference until eviction; an uncacheable handle holds only this reference
      // and is freed here.
      if (filterHandles != null) {
        for (FilterBitsetHandle handle : filterHandles) {
          if (handle != null) handle.decRef();
        }
      }
    }
  }

  // -------------------------------------------------------------------------
  // Per-segment fallback path
  // -------------------------------------------------------------------------

  /**
   * Searches one segment, the hook Lucene drives once per leaf whenever {@link #rewrite} declined
   * the multi-partition path — see that method for the cases. There is no cap on {@code k} here:
   * {@link CuVS2510GPUVectorsReader} runs CAGRA up to k = 1024 and falls to its brute-force index
   * above that, so a large k changes which cuVS index answers the search, not which path Lucene
   * takes.
   */
  @Override
  protected TopDocs approximateSearch(
      LeafReaderContext context,
      Bits acceptDocs,
      int visitedLimit,
      KnnCollectorManager knnCollectorManager)
      throws IOException {
    GPUPerLeafCuVSKnnCollector results =
        new GPUPerLeafCuVSKnnCollector(
            k, visitedLimit, iTopK, searchWidth, threadBlockSize, maxIterations, searchAlgo);
    context.reader().searchNearestVectors(field, getTargetCopy(), results, acceptDocs);
    return results.topDocs();
  }

  // -------------------------------------------------------------------------
  // Filter handle construction
  // -------------------------------------------------------------------------

  /**
   * Returns one {@link FilterBitsetHandle} per segment (in {@code leaves} order) encoding ({@link
   * #filter} ∩ that segment's liveDocs), pulling each from {@link FilterBitsetCache} when the
   * reader state is unchanged. A segment with neither an explicit filter nor deletes gets a {@code
   * null} entry (unfiltered for that partition).
   *
   * <p>The cache key uses the single segment's per-reader key (not just the core key), so liveDocs
   * changes — which happen when a reader is reopened after deletes — invalidate that segment's
   * cached bitset, and an index update only affects the changed segments' entries.
   *
   * <p>Each non-null handle carries a reference the caller must release with {@link
   * FilterBitsetHandle#decRef()} once the search completes.
   *
   * @param segmentAcceptDocs per-segment (filter ∩ liveDocs) masks already evaluated by {@link
   *     #rewrite}, in {@code leaves} order, or {@code null} when the query has no explicit filter
   *     and only deletes need masking
   */
  private List<FilterBitsetHandle> buildPerSegmentFilterHandles(
      List<LeafReaderContext> leaves,
      List<CuVS2510GPUVectorsReader> gpuReaders,
      FixedBitSet[] segmentAcceptDocs)
      throws IOException {

    final int n = leaves.size();
    boolean[] needsFilter = new boolean[n];
    FloatVectorValues[] fvvs = new FloatVectorValues[n];
    long[] entryBytes = new long[n];
    FilterBitsetCache[] caches = new FilterBitsetCache[n];
    Map<FilterBitsetCache, Long> workingSetBytesByCache = new IdentityHashMap<>();
    for (int i = 0; i < n; i++) {
      LeafReaderContext ctx = leaves.get(i);
      // No explicit filter and no deletes in this segment: no filter for this partition.
      if (filter == null && ctx.reader().getLiveDocs() == null) {
        continue;
      }
      needsFilter[i] = true;
      FilterBitsetCache cache = gpuReaders.get(i).getFilterBitsetCache();
      caches[i] = cache;
      FloatVectorValues fvv = gpuReaders.get(i).getFloatVectorValues(field);
      fvvs[i] = fvv;
      // Bytes to hold one bit per vector, word-aligned: this segment's cache-entry size.
      long segBytes = (((long) fvv.size() + 63) / 64) * 8;
      entryBytes[i] = segBytes;
      if (cache.isEnabled()) {
        workingSetBytesByCache.merge(cache, segBytes, Long::sum);
      }
    }

    // Report each cache's complete working set before any acquire so undersizing is diagnosed once.
    for (Map.Entry<FilterBitsetCache, Long> entry : workingSetBytesByCache.entrySet()) {
      entry.getKey().onSearchWorkingSet(entry.getValue());
    }

    List<FilterBitsetHandle> handles = new ArrayList<>(n);
    try {
      for (int i = 0; i < n; i++) {
        if (!needsFilter[i]) {
          handles.add(null);
          continue;
        }
        LeafReaderContext ctx = leaves.get(i);
        FloatVectorValues fvv = fvvs[i];
        FilterBitsetCache cache = caches[i];
        // With an explicit filter the mask is the one already evaluated in rewrite(); without one,
        // this segment is here because it has deletes, so liveDocs alone is the mask.
        final Bits acceptDocs =
            (segmentAcceptDocs != null) ? segmentAcceptDocs[i] : ctx.reader().getLiveDocs();
        // When caching is disabled, route every segment through the uncached, caller-owned path.
        var helper = cache.isEnabled() ? ctx.reader().getReaderCacheHelper() : null;
        if (helper == null) {
          // This reader can't be cached; build an uncached handle owned outright by the caller.
          handles.add(buildSegmentFilterHandle(acceptDocs, fvv));
        } else {
          // Evict entries when this segment reader closes (merge/reopen), not just on LRU.
          cache.ensureCloseListener(helper);
          handles.add(
              cache.acquire(
                  filter,
                  helper.getKey(),
                  field,
                  entryBytes[i],
                  () -> buildSegmentFilterHandle(acceptDocs, fvv)));
        }
      }
      return handles;
    } catch (IOException | RuntimeException | Error e) {
      // The caller cannot release a partially constructed list, so release every acquired handle
      // here before propagating the failure.
      for (FilterBitsetHandle handle : handles) {
        if (handle != null) handle.decRef();
      }
      throw e;
    }
  }

  /**
   * Packs the ordinals of {@code fvv} accepted by {@code acceptDocs} into a new {@link
   * FilterBitsetHandle}. {@code acceptDocs} is this segment's (filter ∩ liveDocs) mask, or its
   * liveDocs alone for a segment with deletes but no explicit Lucene filter; {@code null} follows
   * the Lucene convention that every document is accepted.
   */
  private static FilterBitsetHandle buildSegmentFilterHandle(
      Bits acceptDocs, FloatVectorValues fvv) {
    Bits acceptedOrds = fvv.getAcceptOrds(acceptDocs);
    int numOrds = fvv.size();
    long[] segLongs = new long[(int) (((long) numOrds + 63) / 64)];
    packOrdsToLongs(acceptedOrds, numOrds, segLongs, 0);
    return FilterBitsetHandle.create(segLongs);
  }

  /**
   * Creates the weight for (the explicit filter ∩ the documents that carry a vector for this
   * field), the same conjunction {@code AbstractKnnVectorQuery} filters with. Including {@link
   * FieldExistsQuery} is what makes the resulting cardinality comparable to Lucene's filter cost:
   * a document matching the filter but holding no vector is not a candidate and must not count
   * towards it.
   */
  private Weight createFilterWeight(IndexSearcher indexSearcher) throws IOException {
    BooleanQuery conjunction =
        new BooleanQuery.Builder()
            .add(filter, BooleanClause.Occur.FILTER)
            .add(new FieldExistsQuery(field), BooleanClause.Occur.FILTER)
            .build();
    return indexSearcher.createWeight(
        indexSearcher.rewrite(conjunction), ScoreMode.COMPLETE_NO_SCORES, 1.0f);
  }

  /**
   * Evaluates {@code filterWeight} in {@code ctx}, intersected with that segment's liveDocs, as a
   * materialized bit set over local doc IDs. Materializing it rather than wrapping the filter
   * lazily makes {@link FixedBitSet#cardinality()} — the filter cost the exact-search decision
   * reads — available without a second pass over the filter.
   */
  private static FixedBitSet evalFilter(Weight filterWeight, LeafReaderContext ctx)
      throws IOException {
    FixedBitSet filterBits = new FixedBitSet(ctx.reader().maxDoc());
    ScorerSupplier scorerSupplier = filterWeight.scorerSupplier(ctx);
    if (scorerSupplier == null) {
      // No scorer: the filter matches no documents in this segment.
      return filterBits;
    }
    Bits liveDocs = ctx.reader().getLiveDocs();
    DocIdSetIterator it = scorerSupplier.get(Long.MAX_VALUE).iterator();
    int doc;
    while ((doc = it.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
      if (liveDocs == null || liveDocs.get(doc)) {
        filterBits.set(doc);
      }
    }
    return filterBits;
  }

  /**
   * Packs {@code numOrds} ordinal bits from {@code bits} into {@code dest} starting at long index
   * {@code destLongOffset}. {@code bits == null} means all ordinals accepted (Lucene convention).
   */
  private static void packOrdsToLongs(Bits bits, int numOrds, long[] dest, int destLongOffset) {
    if (bits == null) {
      // All ordinals accepted: fill with all-ones, masking the last partial word.
      int numLongs = (numOrds + 63) / 64;
      Arrays.fill(dest, destLongOffset, destLongOffset + numLongs, -1L);
      int tail = numOrds % 64;
      if (tail != 0) {
        dest[destLongOffset + numLongs - 1] = (1L << tail) - 1L;
      }
      return;
    }
    for (int i = 0; i < numOrds; i++) {
      if (bits.get(i)) {
        dest[destLongOffset + i / 64] |= (1L << (i % 64));
      }
    }
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  /**
   * Unwraps the {@link LeafReaderContext}'s reader to a {@link CuVS2510GPUVectorsReader}, or
   * returns {@code null} if the reader is not of that type.
   */
  private static CuVS2510GPUVectorsReader unwrapGpuReader(LeafReaderContext ctx, String field) {
    var unwrapped = FilterLeafReader.unwrap(ctx.reader());
    if (!(unwrapped instanceof CodecReader)) return null;
    KnnVectorsReader vr = ((CodecReader) unwrapped).getVectorReader();
    // A per-field codec wraps the vectors reader; unwrap to this field's delegate so the
    // multi-partition GPU path is not silently bypassed when the CuVS format is used via
    // PerFieldKnnVectorsFormat.
    if (vr instanceof PerFieldKnnVectorsFormat.FieldsReader perField) {
      vr = perField.getFieldReader(field);
    }
    return (vr instanceof CuVS2510GPUVectorsReader gpuReader) ? gpuReader : null;
  }

  /**
   * Builds a {@link Query} that matches exactly the given pre-scored documents.
   *
   * <p>Partitions {@code scoreDocs} by segment (using {@link ScoreDoc#shardIndex} as the segment
   * offset relative to {@link LeafReaderContext#docBase}), then returns a {@link Scorer} per
   * segment that iterates those docs in ascending doc-ID order and replays their pre-computed
   * scores.
   */
  private static Query docAndScoreQuery(ScoreDoc[] scoreDocs) {
    return new Query() {
      @Override
      public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost)
          throws IOException {
        return new Weight(this) {
          @Override
          public ScorerSupplier scorerSupplier(LeafReaderContext ctx) {
            int base = ctx.docBase;
            int maxDoc = base + ctx.reader().maxDoc();
            // Collect docs belonging to this segment; re-sort by local doc ID ascending.
            int[] localDocs = new int[scoreDocs.length];
            float[] localScores = new float[scoreDocs.length];
            int count = 0;
            float max = Float.NEGATIVE_INFINITY;
            for (ScoreDoc sd : scoreDocs) {
              if (sd.doc >= base && sd.doc < maxDoc) {
                localDocs[count] = sd.doc - base;
                localScores[count] = sd.score * boost;
                if (localScores[count] > max) max = localScores[count];
                count++;
              }
            }
            if (count == 0) return null;
            final int n = count;
            final float maxScore = max;
            Integer[] idx = new Integer[n];
            for (int i = 0; i < n; i++) idx[i] = i;
            Arrays.sort(idx, Comparator.comparingInt(i -> localDocs[i]));
            final int[] sortedDocs = new int[n];
            final float[] sortedScores = new float[n];
            for (int i = 0; i < n; i++) {
              sortedDocs[i] = localDocs[idx[i]];
              sortedScores[i] = localScores[idx[i]];
            }

            return new ScorerSupplier() {
              @Override
              public Scorer get(long leadCost) {
                return new Scorer() {
                  private int pos = -1;

                  @Override
                  public DocIdSetIterator iterator() {
                    return new DocIdSetIterator() {
                      @Override
                      public int docID() {
                        return pos < 0 ? -1 : (pos >= n ? NO_MORE_DOCS : sortedDocs[pos]);
                      }

                      @Override
                      public int nextDoc() {
                        pos++;
                        return docID();
                      }

                      @Override
                      public int advance(int target) {
                        if (pos < 0) pos = 0;
                        while (pos < n && sortedDocs[pos] < target) pos++;
                        return docID();
                      }

                      @Override
                      public long cost() {
                        return n;
                      }
                    };
                  }

                  @Override
                  public float getMaxScore(int upTo) {
                    // The precomputed segment max is a valid upper bound for any upTo.
                    return maxScore;
                  }

                  @Override
                  public float score() {
                    return sortedScores[pos];
                  }

                  @Override
                  public int docID() {
                    return pos < 0
                        ? -1
                        : (pos >= n ? DocIdSetIterator.NO_MORE_DOCS : sortedDocs[pos]);
                  }
                };
              }

              @Override
              public long cost() {
                return n;
              }
            };
          }

          @Override
          public boolean isCacheable(LeafReaderContext ctx) {
            return false;
          }

          @Override
          public Explanation explain(LeafReaderContext ctx, int doc) {
            for (ScoreDoc sd : scoreDocs) {
              if (sd.doc == ctx.docBase + doc) {
                return Explanation.match(
                    sd.score * boost,
                    "GPU multi-segment CAGRA search, product of:",
                    Explanation.match(sd.score, "raw CAGRA score, 1 / (1 + distance)"),
                    Explanation.match(boost, "boost"));
              }
            }
            return Explanation.noMatch("not a GPU search result");
          }
        };
      }

      @Override
      public String toString(String field) {
        return "GPUDocAndScoreQuery";
      }

      @Override
      public void visit(QueryVisitor visitor) {}

      @Override
      public boolean equals(Object o) {
        return this == o;
      }

      @Override
      public int hashCode() {
        return System.identityHashCode(this);
      }
    };
  }
}
