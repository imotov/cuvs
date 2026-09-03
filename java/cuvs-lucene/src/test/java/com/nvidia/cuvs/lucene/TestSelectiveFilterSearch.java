/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.Test;

/**
 * Regression test for the selective-filter strategy {@link
 * org.apache.lucene.search.KnnFloatVectorQuery} documents: a filter leaving at most {@code k}
 * candidate vectors in a segment is answered by an exact scan, so every matching document is
 * returned. {@link GPUKnnFloatVectorQuery} overrides {@code rewrite()}
 * with its own multi-partition CAGRA path, which is approximate; before issue #2522 it ran no
 * matter how selective the filter was, so a filter matching a handful of documents could come back
 * empty.
 *
 * <p>Run against a single-segment and a multi-segment index because the two reach the guarantee
 * differently, and against an index where only some documents carry a vector, which is what makes
 * the filter cost differ from the raw number of filter matches.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestSelectiveFilterSearch extends LuceneTestCase {

  private static final String VECTOR_FIELD = "vectors";
  private static final String ID_FIELD = "id";
  private static final String GROUP_FIELD = "grp";
  private static final String HALF_FIELD = "half";

  private static final int DATASET_SIZE = 600;
  private static final int DIMENSIONS = 32;
  private static final int TOP_K = 10;

  /** Documents per {@link #GROUP_FIELD} value; fewer than {@link #TOP_K}, so a group is selective. */
  private static final int GROUP_SIZE = 6;

  private static final int NUM_GROUPS = DATASET_SIZE / GROUP_SIZE;

  @Test
  public void testSelectiveFilterOnSingleSegment() throws Exception {
    assertSelectiveFiltersAreExact(1);
  }

  @Test
  public void testSelectiveFilterOnMultipleSegments() throws Exception {
    assertSelectiveFiltersAreExact(4);
  }

  private void assertSelectiveFiltersAreExact(int numSegments) throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    try (Directory directory = newDirectory(new ByteBuffersDirectory())) {
      float[][] dataset = generateDataset(random(), DATASET_SIZE, DIMENSIONS);
      try (DirectoryReader reader = buildIndex(directory, dataset, numSegments, false)) {
        assertEquals("unexpected segment count", numSegments, reader.leaves().size());
        IndexSearcher searcher = new IndexSearcher(reader);
        // Every query is for dataset[0]'s neighbourhood, so the matches below are the ones an
        // approximate search is least likely to stumble across on its own.
        float[] target = dataset[0];

        // A filter matching a single document must return that document, however distant it is.
        for (int id : new int[] {1, DATASET_SIZE / 2 + 1, DATASET_SIZE - 1}) {
          assertEquals(
              "filter matching only document " + id,
              Set.of(id),
              search(searcher, reader, target, new TermQuery(new Term(ID_FIELD, "" + id))));
        }

        // GROUP_SIZE (< TOP_K) matches: all of them come back, in whatever order.
        for (int group : new int[] {0, 7, NUM_GROUPS - 1}) {
          Set<Integer> expected = new HashSet<>();
          for (int id = group; id < DATASET_SIZE; id += NUM_GROUPS) {
            expected.add(id);
          }
          assertEquals(GROUP_SIZE, expected.size());
          assertEquals(
              "filter matching group " + group,
              expected,
              search(searcher, reader, target, groupQuery(group)));
        }
      }
    }
  }

  /**
   * The cost that decides between exact and approximate search counts candidate <em>vectors</em>,
   * not filter matches: a document without a vector can never be a hit. This filter matches more
   * than {@link #TOP_K} documents, but only half of them carry a vector, so the search still has to
   * be exact and return all of those.
   */
  @Test
  public void testFilterCostCountsOnlyDocumentsWithVectors() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    final int firstId = 200;
    final int numMatches = TOP_K + 4;

    try (Directory directory = newDirectory(new ByteBuffersDirectory())) {
      float[][] dataset = generateDataset(random(), DATASET_SIZE, DIMENSIONS);
      try (DirectoryReader reader = buildIndex(directory, dataset, 1, true)) {
        IndexSearcher searcher = new IndexSearcher(reader);

        BooleanQuery.Builder filter = new BooleanQuery.Builder();
        Set<Integer> expected = new HashSet<>();
        for (int id = firstId; id < firstId + numMatches; id++) {
          filter.add(new TermQuery(new Term(ID_FIELD, "" + id)), BooleanClause.Occur.SHOULD);
          if (id % 2 == 0) {
            expected.add(id);
          }
        }
        assertTrue("filter should match more than k documents", numMatches > TOP_K);
        assertTrue("but no more than k of them should have a vector", expected.size() <= TOP_K);

        assertEquals(expected, search(searcher, reader, dataset[0], filter.build()));
      }
    }
  }

  /**
   * Just above the exact-search threshold. The filter leaves more than {@code k} candidates, so the
   * multi-partition GPU search does run — but a filter that thin gives CAGRA very little to
   * traverse, and an approximate search that cannot fill {@code k} under a filter has to be redone
   * exactly. That is the last of the three strategies {@code KnnFloatVectorQuery} documents, and
   * without it this returns fewer than {@code k} hits.
   */
  @Test
  public void testFilterJustAboveThresholdStillFillsK() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    try (Directory directory = newDirectory(new ByteBuffersDirectory())) {
      float[][] dataset = generateDataset(random(), DATASET_SIZE, DIMENSIONS);
      try (DirectoryReader reader = buildIndex(directory, dataset, 1, false)) {
        IndexSearcher searcher = new IndexSearcher(reader);

        // TOP_K + 2 documents spread across the id space, so they are neither adjacent in the
        // index nor close together in vector space.
        List<Integer> accepted = new ArrayList<>();
        BooleanQuery.Builder filter = new BooleanQuery.Builder();
        for (int i = 0; i < TOP_K + 2; i++) {
          int id = 13 + i * 50;
          accepted.add(id);
          filter.add(new TermQuery(new Term(ID_FIELD, "" + id)), BooleanClause.Occur.SHOULD);
        }
        assertTrue("filter must clear the exact-search threshold", accepted.size() > TOP_K);

        assertEquals(
            nearest(dataset, dataset[0], accepted, TOP_K),
            search(searcher, reader, dataset[0], filter.build()));
      }
    }
  }

  /**
   * A filter leaving far more than {@code k} candidates keeps using the multi-partition GPU search
   * — the deferral to Lucene is meant for selective filters only, not a blanket opt-out. The
   * rewritten query identifies which path ran: the GPU one produces its own doc-and-score query.
   */
  @Test
  public void testBroadFilterStillUsesMultiPartitionSearch() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    try (Directory directory = newDirectory(new ByteBuffersDirectory())) {
      float[][] dataset = generateDataset(random(), DATASET_SIZE, DIMENSIONS);
      try (DirectoryReader reader = buildIndex(directory, dataset, 4, false)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        Query filter = new TermQuery(new Term(HALF_FIELD, "h0"));

        Query rewritten = query(dataset[0], filter).rewrite(searcher);
        assertEquals(
            "a broad filter should not fall back to the per-segment path",
            "GPUDocAndScoreQuery",
            rewritten.toString(VECTOR_FIELD));

        // The broad filter is still a filter: nothing outside it may be returned.
        for (int id : search(searcher, reader, dataset[0], filter)) {
          assertEquals("returned a document outside the filter: " + id, 0, id % 2);
        }
      }
    }
  }

  /** The {@code k} ids among {@code candidates} whose vectors are nearest to {@code target}. */
  private static Set<Integer> nearest(
      float[][] dataset, float[] target, List<Integer> candidates, int k) {
    return candidates.stream()
        .sorted(Comparator.comparingDouble(id -> squaredDistance(target, dataset[id])))
        .limit(k)
        .collect(Collectors.toCollection(HashSet::new));
  }

  private static double squaredDistance(float[] a, float[] b) {
    double sum = 0;
    for (int i = 0; i < a.length; i++) {
      double d = a[i] - b[i];
      sum += d * d;
    }
    return sum;
  }

  private static Query groupQuery(int group) {
    return new TermQuery(new Term(GROUP_FIELD, "g" + group));
  }

  private static GPUKnnFloatVectorQuery query(float[] target, Query filter) {
    return new GPUKnnFloatVectorQuery(VECTOR_FIELD, target, TOP_K, filter, TOP_K, 1);
  }

  /** Runs a filtered GPU kNN search and returns the ids of the documents it matched. */
  private static Set<Integer> search(
      IndexSearcher searcher, IndexReader reader, float[] target, Query filter) throws IOException {
    ScoreDoc[] hits = searcher.search(query(target, filter), TOP_K).scoreDocs;
    Set<Integer> ids = new HashSet<>();
    for (ScoreDoc hit : hits) {
      ids.add(Integer.parseInt(reader.storedFields().document(hit.doc).get(ID_FIELD)));
    }
    assertEquals("duplicate documents in the results", hits.length, ids.size());
    return ids;
  }

  /**
   * Indexes {@link #DATASET_SIZE} documents split evenly across {@code numSegments} segments, with
   * merging disabled so the commits survive as segments. When {@code sparseVectors} is set only
   * even-numbered documents carry a vector, leaving the rest as filter matches that can never be
   * hits.
   */
  private static DirectoryReader buildIndex(
      Directory directory, float[][] dataset, int numSegments, boolean sparseVectors)
      throws Exception {
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(new CuVS2510GPUSearchCodec())
            .setMergePolicy(NoMergePolicy.INSTANCE);
    try (IndexWriter writer = new IndexWriter(directory, config)) {
      final int commitEvery = DATASET_SIZE / numSegments;
      for (int i = 0; i < DATASET_SIZE; i++) {
        Document doc = new Document();
        doc.add(new StringField(ID_FIELD, String.valueOf(i), Field.Store.YES));
        doc.add(new StringField(GROUP_FIELD, "g" + (i % NUM_GROUPS), Field.Store.NO));
        doc.add(new StringField(HALF_FIELD, "h" + (i % 2), Field.Store.NO));
        if (!sparseVectors || i % 2 == 0) {
          doc.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
        }
        writer.addDocument(doc);
        if ((i + 1) % commitEvery == 0) {
          writer.commit();
        }
      }
    }
    return DirectoryReader.open(directory);
  }
}
