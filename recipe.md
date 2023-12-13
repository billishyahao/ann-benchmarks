## ANN-Benchmark-Recipe
[ANN-Benchmark](https://ann-benchmarks.com/) is a benchmarking environment for approximate nearest neighbor algorithms search. This benchmark contains tools to benchmark various implementations of approximate nearest neighbor (ANN) search for selected metrics. We have pre-generated datasets (in HDF5 format) and prepared Docker containers for each algorithm, as well as a [test suite](https://github.com/erikbern/ann-benchmarks/actions) to verify function integrity.

## Problem Definition
Doing fast searching of nearest neighbors in high dimensional spaces is an increasingly important problem with notably few empirical attempts at comparing approaches in an objective way, despite a clear need for such to drive optimization forward.

## Keywords
Data mining, Information Processing

## Author



## Install
The only prerequisite is Python (tested with 3.10.6) and Docker.

1. Clone the repo.
2. Run `pip install -r requirements.txt`.
3. Run `python install.py` to build all the libraries inside Docker containers (this can take a while, like 10-30 minutes).



## Benchmark
### Prerequsitions
1. Disable CStates
   ```shell
   cpupower idle-set -d 3
   cpupower idle-set -d 2
   ```
2. set performance mode of frequency governor
   ```shell
   cpupower frequency-set -g performance
   cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
   ```

### Run.py commandline explanation

```
usage: run.py [-h] [--dataset NAME] [-k COUNT] [--definitions FOLDER] [--algorithm NAME] [--docker-tag NAME] [--list-algorithms] [--force] [--runs COUNT] [--timeout TIMEOUT] [--local] [--batch] [--max-n-algorithms MAX_N_ALGORITHMS] [--run-disabled] [--parallelism PARALLELISM]

options:
  -h, --help            show this help message and exit
  --dataset NAME        the dataset to load training points from (default: glove-100-angular)
  -k COUNT, --count COUNT
                        the number of near neighbours to search for (default: 10)
  --definitions FOLDER  base directory of algorithms. Algorithm definitions expected at 'FOLDER/*/config.yml' (default: ann_benchmarks/algorithms)
  --algorithm NAME      run only the named algorithm (default: None)
  --docker-tag NAME     run only algorithms in a particular docker image (default: None)
  --list-algorithms     print the names of all known algorithms and exit (default: False)
  --force               re-run algorithms even if their results already exist (default: False)
  --runs COUNT          run each algorithm instance COUNT times and use only the best result (default: 5)
  --timeout TIMEOUT     Timeout (in seconds) for each individual algorithm run, or -1if no timeout should be set (default: 7200)
  --local               If set, then will run everything locally (inside the same process) rather than using Docker (default: False)
  --batch               If set, algorithms get all queries at once (default: False)
  --max-n-algorithms MAX_N_ALGORITHMS
                        Max number of algorithms to run (just used for testing) (default: -1)
  --run-disabled        run algorithms that are disabled in algos.yml (default: False)
  --parallelism PARALLELISM
                        Number of Docker containers in parallel (default: 1)
```

### Benchmark for local environment (bypass docker network issue!!!)
1. Check that `ann_benchmarks/algorithms/{ALGORITHM YOU WANT TO TEST}/Dockerfile` contains the docker environment that you want to test and pip manually to install required python library.
1. Check that `ann_benchmarks/algorithms/{ALGORITHM YOU WANT TO TEST}/config.yml` contains the parameter settings that you want to test
ann_benchmarks/algorithms/scann/Dockerfile
1. Run `python run.py --algorithm {ALGORITHM YOU WANT TO TEST} --local`
2. Run `python plot.py` or `python create_website.py` to plot results.
3. Run `python data_export.py --out res.csv` to export all results into a csv file for additional post-processing.

 

### Benchmark with SCANN(glove-100-angular) instance

1. Modify config.yml to contain your customized parameters.
```
diff --git a/ann_benchmarks/algorithms/scann/config.yml b/ann_benchmarks/algorithms/scann/config.yml
index 728dc96..c648847 100644
--- a/ann_benchmarks/algorithms/scann/config.yml
+++ b/ann_benchmarks/algorithms/scann/config.yml
@@ -8,28 +8,149 @@ float:
     name: scann
     run_groups:
       scann1:
-        args: [[2000], [0.2], [2], [dot_product]]
-        query_args: [[[1, 30], [2, 30], [4, 30], [8, 30], [30, 120], [35, 100], [
-              40, 80], [45, 80], [50, 80], [55, 95], [60, 110], [65, 110], [75, 110],
-            [90, 110], [110, 120], [130, 150], [150, 200], [170, 200], [200, 300],
-            [220, 500], [250, 500], [310, 300], [400, 300], [500, 500], [800, 1000]]]
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
       scann2:
         args: [[1500], [0.55], [1], [dot_product]]
-        query_args: [[[1, 30], [2, 30], [4, 30], [8, 30], [8, 25], [10, 25], [12,
-              25], [13, 25], [14, 27], [15, 30], [17, 30], [18, 40], [20, 40], [22,
-              40], [25, 50], [30, 50], [35, 55], [50, 60], [60, 60], [80, 80], [100,
-              100]]]
+        query_args: [[[1500, 1000]]]
       scann3:
-        args: [[1000], [0.2], [1], [dot_product]]
-        query_args: [[[1, 30], [2, 30], [4, 30], [8, 30], [9, 25], [11, 35], [12,
-              35], [13, 35], [14, 40], [15, 40], [16, 40], [17, 45], [20, 45], [20,
-              55], [25, 55], [25, 70], [30, 70], [40, 90], [50, 100], [60, 120], [
-              70, 140]]]
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
       scann4:
-        args: [[1400], [0.15], [3], [dot_product]]
-        query_args: [[[1, 30], [4, 30], [9, 30], [16, 32], [25, 50], [36, 72], [49,
-              98], [70, 150], [90, 200], [120, 210], [180, 270], [210, 330], [260,
-              400], [320, 500], [400, 600], [500, 700], [800, 900]]]
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann5:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann6:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann7:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann8:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann9:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann10:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann11:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann12:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann13:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann14:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann15:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann16:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann17:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann18:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann19:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann20:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann21:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann22:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann23:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann24:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann25:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann26:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann27:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann28:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann29:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann30:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann31:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann32:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann33:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann34:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann35:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann36:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann37:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann38:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann39:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann40:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann41:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann42:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann43:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann44:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann45:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann46:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
+      scann47:
+        args: [[2000], [0.2], [1], [dot_product]]
+        query_args: [[[2000, 1000]]]
+      scann48:
+        args: [[1500], [0.55], [1], [dot_product]]
+        query_args: [[[1500, 1000]]]
   euclidean:
   - base_args: {}
     constructor: Scann
```

2. run the algorithm:
numactl -C 0-55 python run.py --algorithm scann --local -k 10 --parallelism 56

3. use plot.py to collect performance graph and the results will be saved into results/glove-100-angular.png



## Result Check
Check performance data in graph such as queries per second under different recall constrained.

## More infos.

### Algorithm supported

* [Annoy](https://github.com/spotify/annoy) ![https://img.shields.io/github/stars/spotify/annoy?style=social](https://img.shields.io/github/stars/spotify/annoy?style=social)
* [FLANN](http://www.cs.ubc.ca/research/flann/) ![https://img.shields.io/github/stars/flann-lib/flann?style=social](https://img.shields.io/github/stars/flann-lib/flann?style=social)
* [scikit-learn](http://scikit-learn.org/stable/modules/neighbors.html): LSHForest, KDTree, BallTree
* [Weaviate](https://github.com/weaviate/weaviate) ![https://img.shields.io/github/stars/weaviate/weaviate?style=social](https://img.shields.io/github/stars/weaviate/weaviate?style=social)
* [PANNS](https://github.com/ryanrhymes/panns) ![https://img.shields.io/github/stars/ryanrhymes/panns?style=social](https://img.shields.io/github/stars/ryanrhymes/panns?style=social)
* [NearPy](http://pixelogik.github.io/NearPy/) ![https://img.shields.io/github/stars/pixelogik/NearPy?style=social](https://img.shields.io/github/stars/pixelogik/NearPy?style=social)
* [KGraph](https://github.com/aaalgo/kgraph) ![https://img.shields.io/github/stars/aaalgo/kgraph?style=social](https://img.shields.io/github/stars/aaalgo/kgraph?style=social)
* [NMSLIB (Non-Metric Space Library)](https://github.com/nmslib/nmslib) ![https://img.shields.io/github/stars/nmslib/nmslib?style=social](https://img.shields.io/github/stars/nmslib/nmslib?style=social): SWGraph, HNSW, BallTree, MPLSH
* [hnswlib (a part of nmslib project)](https://github.com/nmslib/hnsw) ![https://img.shields.io/github/stars/nmslib/hnsw?style=social](https://img.shields.io/github/stars/nmslib/hnsw?style=social)
* [RPForest](https://github.com/lyst/rpforest) ![https://img.shields.io/github/stars/lyst/rpforest?style=social](https://img.shields.io/github/stars/lyst/rpforest?style=social)
* [FAISS](https://github.com/facebookresearch/faiss) ![https://img.shields.io/github/stars/facebookresearch/faiss?style=social](https://img.shields.io/github/stars/facebookresearch/faiss?style=social)
* [DolphinnPy](https://github.com/ipsarros/DolphinnPy) ![https://img.shields.io/github/stars/ipsarros/DolphinnPy?style=social](https://img.shields.io/github/stars/ipsarros/DolphinnPy?style=social)
* [Datasketch](https://github.com/ekzhu/datasketch) ![https://img.shields.io/github/stars/ekzhu/datasketch?style=social](https://img.shields.io/github/stars/ekzhu/datasketch?style=social)
* [nndescent](https://github.com/brj0/nndescent) ![https://img.shields.io/github/stars/brj0/nndescent?style=social](https://img.shields.io/github/stars/brj0/nndescent?style=social)
* [PyNNDescent](https://github.com/lmcinnes/pynndescent) ![https://img.shields.io/github/stars/lmcinnes/pynndescent?style=social](https://img.shields.io/github/stars/lmcinnes/pynndescent?style=social)
* [MRPT](https://github.com/teemupitkanen/mrpt) ![https://img.shields.io/github/stars/teemupitkanen/mrpt?style=social](https://img.shields.io/github/stars/teemupitkanen/mrpt?style=social)
* [NGT](https://github.com/yahoojapan/NGT) ![https://img.shields.io/github/stars/yahoojapan/NGT?style=social](https://img.shields.io/github/stars/yahoojapan/NGT?style=social): ONNG, PANNG, QG
* [SPTAG](https://github.com/microsoft/SPTAG) ![https://img.shields.io/github/stars/microsoft/SPTAG?style=social](https://img.shields.io/github/stars/microsoft/SPTAG?style=social)
* [PUFFINN](https://github.com/puffinn/puffinn) ![https://img.shields.io/github/stars/puffinn/puffinn?style=social](https://img.shields.io/github/stars/puffinn/puffinn?style=social)
* [N2](https://github.com/kakao/n2) ![https://img.shields.io/github/stars/kakao/n2?style=social](https://img.shields.io/github/stars/kakao/n2?style=social)
* [ScaNN](https://github.com/google-research/google-research/tree/master/scann)
* [Vearch](https://github.com/vearch/vearch) ![https://img.shields.io/github/stars/vearch/vearch?style=social](https://img.shields.io/github/stars/vearch/vearch?style=social)
* [Elasticsearch](https://github.com/elastic/elasticsearch) ![https://img.shields.io/github/stars/elastic/elasticsearch?style=social](https://img.shields.io/github/stars/elastic/elasticsearch?style=social): HNSW
* [Elastiknn](https://github.com/alexklibisz/elastiknn) ![https://img.shields.io/github/stars/alexklibisz/elastiknn?style=social](https://img.shields.io/github/stars/alexklibisz/elastiknn?style=social)
* [OpenSearch KNN](https://github.com/opensearch-project/k-NN) ![https://img.shields.io/github/stars/opensearch-project/k-NN?style=social](https://img.shields.io/github/stars/opensearch-project/k-NN?style=social)
* [DiskANN](https://github.com/microsoft/diskann) ![https://img.shields.io/github/stars/microsoft/diskann?style=social](https://img.shields.io/github/stars/microsoft/diskann?style=social): Vamana, Vamana-PQ
* [Vespa](https://github.com/vespa-engine/vespa) ![https://img.shields.io/github/stars/vespa-engine/vespa?style=social](https://img.shields.io/github/stars/vespa-engine/vespa?style=social)
* [scipy](https://docs.scipy.org/doc/scipy/reference/spatial.html): cKDTree
* [vald](https://github.com/vdaas/vald) ![https://img.shields.io/github/stars/vdaas/vald?style=social](https://img.shields.io/github/stars/vdaas/vald?style=social)
* [Qdrant](https://github.com/qdrant/qdrant) ![https://img.shields.io/github/stars/qdrant/qdrant?style=social](https://img.shields.io/github/stars/qdrant/qdrant?style=social)
* [HUAWEI(qsgngt)](https://github.com/WPJiang/HWTL_SDU-ANNS.git)
* [Milvus](https://github.com/milvus-io/milvus) ![https://img.shields.io/github/stars/milvus-io/milvus?style=social](https://img.shields.io/github/stars/milvus-io/milvus?style=social): [Knowhere](https://github.com/milvus-io/knowhere)
* [Zilliz(Glass)](https://github.com/hhy3/pyglass)
* [pgvector](https://github.com/pgvector/pgvector) ![https://img.shields.io/github/stars/pgvector/pgvector?style=social](https://img.shields.io/github/stars/pgvector/pgvector?style=social)
* [RediSearch](https://github.com/redisearch/redisearch) ![https://img.shields.io/github/stars/redisearch/redisearch?style=social](https://img.shields.io/github/stars/redisearch/redisearch?style=social)
  * [pg_embedding](https://github.com/neondatabase/pg_embedding) ![https://img.shields.io/github/stars/pg_embedding/pg_embedding?style=social](https://img.shields.io/github/stars/neondatabase/pg_embedding?style=social)

### Data sets supported

We have a number of precomputed data sets in HDF5 format. All data sets have been pre-split into train/test and include ground truth data for the top-100 nearest neighbors.

| Dataset                                                           | Dimensions | Train size | Test size | Neighbors | Distance  | Download                                                                   |
| ----------------------------------------------------------------- | ---------: | ---------: | --------: | --------: | --------- | -------------------------------------------------------------------------- |
| [DEEP1B](http://sites.skoltech.ru/compvision/noimi/)              |         96 |  9,990,000 |    10,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/deep-image-96-angular.hdf5) (3.6GB)
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) |        784 |     60,000 |    10,000 |       100 | Euclidean | [HDF5](http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5) (217MB) |
| [GIST](http://corpus-texmex.irisa.fr/)                            |        960 |  1,000,000 |     1,000 |       100 | Euclidean | [HDF5](http://ann-benchmarks.com/gist-960-euclidean.hdf5) (3.6GB)          |
| [GloVe](http://nlp.stanford.edu/projects/glove/)                  |         25 |  1,183,514 |    10,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/glove-25-angular.hdf5) (121MB)            |
| GloVe                                                             |         50 |  1,183,514 |    10,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/glove-50-angular.hdf5) (235MB)            |
| GloVe                                                             |        100 |  1,183,514 |    10,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/glove-100-angular.hdf5) (463MB)           |
| GloVe                                                             |        200 |  1,183,514 |    10,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/glove-200-angular.hdf5) (918MB)           |
| [Kosarak](http://fimi.uantwerpen.be/data/)                        |      27,983 |     74,962 |       500 |       100 | Jaccard   | [HDF5](http://ann-benchmarks.com/kosarak-jaccard.hdf5) (33MB)             |
| [MNIST](http://yann.lecun.com/exdb/mnist/)                        |        784 |     60,000 |    10,000 |       100 | Euclidean | [HDF5](http://ann-benchmarks.com/mnist-784-euclidean.hdf5) (217MB)         |
| [MovieLens-10M](https://grouplens.org/datasets/movielens/10m/)  |      65,134 |     69,363 |       500 |       100 | Jaccard   | [HDF5](http://ann-benchmarks.com/movielens10m-jaccard.hdf5) (63MB)             |
| [NYTimes](https://archive.ics.uci.edu/ml/datasets/bag+of+words)   |        256 |    290,000 |    10,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/nytimes-256-angular.hdf5) (301MB)         |
| [SIFT](http://corpus-texmex.irisa.fr/)                           |        128 |  1,000,000 |    10,000 |       100 | Euclidean | [HDF5](http://ann-benchmarks.com/sift-128-euclidean.hdf5) (501MB)          |
| [Last.fm](https://github.com/erikbern/ann-benchmarks/pull/91)     |         65 |    292,385 |    50,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/lastfm-64-dot.hdf5) (135MB)               |


