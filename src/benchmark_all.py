import time
import gc
import os
import sys
import glob
import shutil
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta

# ==========================================
# 0. 環境設定 & ライブラリ検出
# ==========================================
# 計測するデータサイズ (行数)
DATA_SIZES = [
    1_000_000,       # 1M
    10_000_000,      # 10M
    100_000_000,     # 100M
    1_000_000_000,   # 1B
    # 10_000_000_000 # 10B
]

BASE_DIR = "./bench_data"
os.makedirs(BASE_DIR, exist_ok=True)

# 結果格納用のリスト
ALL_RESULTS = []

# ライブラリのインポート（存在チェック）
try: import cudf; import cupy as cp
except ImportError: cudf = None

try: import duckdb
except ImportError: duckdb = None

try: import dask.dataframe as dd; from dask.distributed import Client, LocalCluster
except ImportError: dd = None

import dask_cudf; from dask_cuda import LocalCUDACluster
try: import dask_cudf; from dask_cuda import LocalCUDACluster
except ImportError: dask_cudf = None

try: from pyspark.sql import SparkSession; from pyspark.sql import functions as F; from pyspark.sql.window import Window
except ImportError: SparkSession = None

print(f"Libraries Detected: Pandas, Polars, "
      f"cuDF={cudf is not None}, DuckDB={duckdb is not None}, "
      f"Dask={dd is not None}, Spark={SparkSession is not None}")

current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.dirname(current_dir)
if os.path.exists(os.path.join(project_root, "pyproject.toml")):
    os.chdir(project_root)

# 2. PySpark 対策: Java 17 の自動設定
#    (優先順位: プロジェクト内の.jdkフォルダ -> システムのJava 17 -> システムのJava 11)
jdk_candidates = [os.path.join(project_root, ".jdk")] + \
                 sorted(glob.glob("/usr/lib/jvm/java-17*")) + \
                 sorted(glob.glob("/usr/lib/jvm/java-11*"))

found_java = False
for candidate in jdk_candidates:
    # bin/java が存在するかチェック
    if os.path.exists(os.path.join(candidate, "bin", "java")):
        # 環境変数を上書き
        os.environ["JAVA_HOME"] = candidate
        os.environ["PATH"] = os.path.join(candidate, "bin") + ":" + os.environ.get("PATH", "")
        print(f"✅ Using Java: {candidate}")
        found_java = True
        break

if not found_java:
    print("⚠️  Java 17/11 not found. PySpark might fail (requires Java 17).")

# 3. GPUライブラリ (cuDF/Dask) 対策: NVIDIAライブラリパスの自動追加
libs = glob.glob(os.path.join(sys.prefix, "lib", "python*", "site-packages", "nvidia", "*", "lib"))
if libs:
    current_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_libs = [lib for lib in libs if lib not in current_path]
    if new_libs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(new_libs) + ":" + current_path

# ==========================================
# 1. データ生成 (Parquet Generator)
# ==========================================
def generate_parquet(n_rows, chunk_size=10_000_000):
    data_dir = os.path.join(BASE_DIR, f"data_{n_rows}")
    join_file = os.path.join(BASE_DIR, f"join_{n_rows}.parquet")
    
    if os.path.exists(data_dir) and os.path.exists(join_file):
        if len(glob.glob(os.path.join(data_dir, "*.parquet"))) > 0:
            print(f"[Gen] Data for {n_rows:,} rows found. Skipping generation.")
            return data_dir, join_file

    print(f"[Gen] Generating {n_rows:,} rows into {data_dir} ...")
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    n_cats = min(max(n_rows // 100, 10_000), 1_000_000)
    pl.DataFrame({
        "id": np.arange(n_cats, dtype=np.int32),
        "master_score": np.random.rand(n_cats).astype(np.float32)
    }).write_parquet(join_file)

    rows_generated = 0
    part_idx = 0
    start_date = datetime(2023, 1, 1)

    while rows_generated < n_rows:
        current_chunk = min(chunk_size, n_rows - rows_generated)
        end_date = start_date + timedelta(seconds=current_chunk - 1)
        
        df = pl.DataFrame({
            "id": np.random.randint(0, n_cats, current_chunk, dtype=np.int32),
            "category": np.random.choice(['Alpha', 'Beta', 'Gamma', 'Delta'], current_chunk),
            "v1": np.random.rand(current_chunk).astype(np.float32),
            "v2": np.random.randn(current_chunk).astype(np.float32) * 100,
            "text_col": np.random.choice([
                "Error: Code 404 Not Found", "Info: Status 200 OK", 
                "Warning: Disk Full 90%", "Critical: System Failure 500"
            ], current_chunk),
            "date": pl.datetime_range(
                start=start_date,
                end=end_date,
                interval="1s",
                eager=True
            ).cast(pl.Datetime)
        })
        
        file_path = os.path.join(data_dir, f"part_{part_idx:05d}.parquet")
        df.write_parquet(file_path)
        
        rows_generated += current_chunk
        part_idx += 1
        start_date = end_date + timedelta(seconds=1)

        if part_idx % 5 == 0:
            print(f"  ... {rows_generated:,} rows")
        
        del df
        gc.collect()

    return data_dir, join_file

# ==========================================
# 2. ベンチマークランナー
# ==========================================
class Runner:
    def __init__(self, n_rows, data_dir, join_file):
        self.n_rows = n_rows
        self.data_dir = data_dir
        self.join_file = join_file
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))

    def measure(self, name, task, func):
        gc.collect()
        time.sleep(0.5)
        print(f"  -> [{name}] {task} ... ", end="", flush=True)
        duration = None
        status = "Success"
        try:
            start = time.perf_counter()
            func()
            duration = time.perf_counter() - start
            print(f"{duration:.4f} sec")
        except MemoryError:
            print("OOM")
            status = "OOM"
        except Exception as e:
            print(f"Failed: {str(e)[:50]}")
            status = "Failed"
        
        # 結果をリストに追加
        ALL_RESULTS.append({
            "library": name,
            "rows": self.n_rows,
            "task": task,
            "duration_sec": duration if duration else np.nan,
            "status": status
        })

    # ----------------------------------------------------
    # 1. Pandas
    # ----------------------------------------------------
    def run_pandas(self):
        if self.n_rows > 100_000_000:
            print("  [Pandas] Skipped (>100M rows)")
            return
        
        try:
            df = pd.concat([pd.read_parquet(f) for f in self.files], ignore_index=True)
            df_join = pd.read_parquet(self.join_file)
        except:
            print("  [Pandas] Load Failed (OOM)")
            return

        self.measure("Pandas", "T1:Math", lambda: (
            np.log(df["v1"]) + np.sin(df["v2"] * np.pi) * np.sqrt(df["v1"]**2)
        ))
        self.measure("Pandas", "T2:Regex", lambda: (
            df[df["text_col"].str.extract(r"(\d+)", expand=False).astype(float).fillna(0) >= 400]
        ))
        self.measure("Pandas", "T3:Join", lambda: (
            pd.merge(df, df_join, on="id", how="left")
        ))
        self.measure("Pandas", "T4:Agg", lambda: (
            df.groupby("id").agg({"v1": "mean", "v2": "sum"})
        ))
        self.measure("Pandas", "T5:Case", lambda: (
            np.select([(df["v1"]>0.8)&(df["category"]=='Alpha')], ['High'], default='Normal')
        ))
        self.measure("Pandas", "T6:Roll", lambda: (
            df.sort_values("date").rolling(1000, on="date")["v1"].mean()
        ))
        del df, df_join

    # ----------------------------------------------------
    # 2 & 3. Polars (CPU/GPU)
    # ----------------------------------------------------
    def run_polars(self, engine):
        lf = pl.scan_parquet(os.path.join(self.data_dir, "*.parquet"))
        lf_join = pl.scan_parquet(self.join_file)
        name = f"Polars({engine})"

        def run(q):
            q.collect(engine=engine)

        # T1: Math
        q1 = lf.select((pl.col("v1").log() + (pl.col("v2")*np.pi).sin() * (pl.col("v1")**2).sqrt()).alias("res")).select(pl.col("res").sum())
        self.measure(name, "T1:Math", lambda: run(q1))

        # T2: Regex
        q2 = lf.filter(
            pl.col("text_col").str.extract(r"(\d+)", 1).cast(pl.Int32, strict=False) >= 400
        ).select(pl.len())
        self.measure(name, "T2:Regex", lambda: run(q2))

        # T3: Join
        q3 = lf.join(lf_join, on="id", how="left").select((pl.col("v1")*pl.col("master_score")).sum())
        self.measure(name, "T3:Join", lambda: run(q3))

        # T4: Agg
        q4 = lf.group_by("id").agg([pl.col("v1").mean(), pl.col("v2").sum()])
        self.measure(name, "T4:Agg", lambda: q4.collect(engine=engine))

        # T5: Case
        q5 = lf.select(
            pl.when((pl.col("v1")>0.8)&(pl.col("category")=='Alpha')).then(1).otherwise(0).alias("f")
        ).select(pl.col("f").sum())
        self.measure(name, "T5:Case", lambda: run(q5))

        # T6: Roll
        # rolling_meanを使用 (バージョン互換性)
        q6 = lf.sort("date").select(pl.col("v1").rolling_mean(1000).sum())
        self.measure(name, "T6:Roll", lambda: run(q6))

    # ----------------------------------------------------
    # 4. cuDF (GPU)
    # ----------------------------------------------------
    def run_cudf(self):
        if not cudf: return
        if self.n_rows > 500_000_000:
            print("  [cuDF] Skipped (>500M rows)")
            return

        try:
            df = cudf.read_parquet(self.files)
            df_join = cudf.read_parquet(self.join_file)
        except:
            print("  [cuDF] Load Failed (OOM)")
            return

        self.measure("cuDF", "T1:Math", lambda: (
            cp.log(df["v1"]) + cp.sin(df["v2"] * cp.pi) * cp.sqrt(df["v1"]**2)
        ))
        self.measure("cuDF", "T2:Regex", lambda: (
            df[df["text_col"].str.extract(r"(\d+)")[0].astype(float) >= 400]
        ))
        self.measure("cuDF", "T3:Join", lambda: (
            df.merge(df_join, on="id", how="left")
        ))
        self.measure("cuDF", "T4:Agg", lambda: (
            df.groupby("id").agg({"v1": "mean", "v2": "sum"})
        ))
        self.measure("cuDF", "T5:Case", lambda: (
            df["v1"].copy() 
        ))
        # on="date" を削除
        self.measure("cuDF", "T6:Roll", lambda: (
            df.sort_values("date").rolling(1000)["v1"].mean()
        ))
        del df, df_join

    # ----------------------------------------------------
    # 5. DuckDB
    # ----------------------------------------------------
    def run_duckdb(self):
        if not duckdb: return
        con = duckdb.connect()
        con.execute(f"CREATE OR REPLACE VIEW t_main AS SELECT * FROM '{os.path.join(self.data_dir, '*.parquet')}'")
        con.execute(f"CREATE OR REPLACE VIEW t_join AS SELECT * FROM '{self.join_file}'")
        
        def q(sql): con.execute(sql).fetchall()

        self.measure("DuckDB", "T1:Math", lambda: q(
            "SELECT sum(ln(v1) + sin(v2 * pi()) * sqrt(pow(v1, 2))) FROM t_main"
        ))
        self.measure("DuckDB", "T2:Regex", lambda: q(
            "SELECT count(*) FROM t_main WHERE CAST(regexp_extract(text_col, '(\\d+)', 1) AS INT) >= 400"
        ))
        self.measure("DuckDB", "T3:Join", lambda: q(
            "SELECT sum(t1.v1 * t2.master_score) FROM t_main t1 LEFT JOIN t_join t2 ON t1.id = t2.id"
        ))
        self.measure("DuckDB", "T4:Agg", lambda: q(
            "SELECT id, avg(v1), sum(v2) FROM t_main GROUP BY id"
        ))
        self.measure("DuckDB", "T5:Case", lambda: q(
            "SELECT sum(CASE WHEN v1>0.8 AND category='Alpha' THEN 1 ELSE 0 END) FROM t_main"
        ))
        # サブクエリを使用
        self.measure("DuckDB", "T6:Roll", lambda: q(
            """
            SELECT sum(x) FROM (
                SELECT avg(v1) OVER (ORDER BY date ROWS BETWEEN 999 PRECEDING AND CURRENT ROW) as x 
                FROM t_main
            )
            """
        ))

    # ----------------------------------------------------
    # 6 & 7. Dask (CPU/GPU)
    # ----------------------------------------------------
    def run_dask(self, gpu=False):
        lib = dask_cudf if gpu else dd
        if not lib: return
        name = f"Dask({'GPU' if gpu else 'CPU'})"

        try:
            df = lib.read_parquet(os.path.join(self.data_dir, "*.parquet"))
            df_join = lib.read_parquet(self.join_file)
        except: return

        def run(task): task.compute()

        self.measure(name, "T1:Math", lambda: run(
            (np.log(df["v1"]) + np.sin(df["v2"] * np.pi) * np.sqrt(df["v1"]**2)).sum()
        ))
        self.measure(name, "T2:Regex", lambda: run(
            df[df["text_col"].str.extract(r"(\d+)")[0].astype(float) >= 400].shape[0]
        ))
        self.measure(name, "T3:Join", lambda: run(
            df.merge(df_join, on="id", how="left")["v1"].sum()
        ))
        self.measure(name, "T4:Agg", lambda: run(
            df.groupby("id").agg({"v1": "mean", "v2": "sum"})
        ))
        self.measure(name, "T5:Case", lambda: run(
            df[df["v1"] > 0.8].shape[0] 
        ))
        self.measure(name, "T6:Roll", lambda: run(
            df.map_partitions(lambda d: d.sort_values("date")["v1"].rolling(1000).mean()).sum()
        ))

    # ----------------------------------------------------
    # 8. PySpark
    # ----------------------------------------------------
    def run_pyspark(self):
        if not SparkSession: return
        
        try:
            spark = SparkSession.builder.appName("Bench")\
                .config("spark.driver.memory", "32g")\
                .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
                .getOrCreate()
            spark.sparkContext.setLogLevel("ERROR")
        except Exception as e:
            print(f"  [Spark] Init Failed: {str(e)[:100]}")
            return
        
        try:
            sdf = spark.read.parquet(os.path.join(self.data_dir, "*.parquet"))
            sdf_join = spark.read.parquet(self.join_file)
            
            def run(df): df.write.format("noop").mode("overwrite").save()

            self.measure("Spark", "T1:Math", lambda: run(
                sdf.select(F.log("v1") + F.sin(F.col("v2")*np.pi) * F.sqrt(F.pow("v1", 2)))
            ))
            self.measure("Spark", "T2:Regex", lambda: run(
                sdf.filter(F.regexp_extract("text_col", r"(\d+)", 1).cast("int") >= 400)
            ))
            self.measure("Spark", "T3:Join", lambda: run(
                sdf.join(sdf_join, "id", "left")
            ))
            self.measure("Spark", "T4:Agg", lambda: run(
                sdf.groupBy("id").agg(F.mean("v1"), F.sum("v2"))
            ))
            self.measure("Spark", "T5:Case", lambda: run(
                sdf.withColumn("f", F.when((F.col("v1")>0.8), 1).otherwise(0))
            ))
            self.measure("Spark", "T6:Roll", lambda: run(
                sdf.withColumn("roll", F.mean("v1").over(
                    Window.partitionBy("id").orderBy("date").rowsBetween(-999,0)
                ))
            ))
        finally:
            spark.stop()

# ==========================================
# メイン実行
# ==========================================
if __name__ == "__main__":
    client = None
    if dask_cudf:
        try:
            cluster = LocalCUDACluster()
            client = Client(cluster)
            print(f"Dask GPU Cluster: {client.dashboard_link}")
        except: pass
    elif dd:
        client = Client()

    for n_rows in DATA_SIZES:
        print(f"\n\n############################################")
        print(f"  SIZE: {n_rows:,} ROWS")
        print(f"############################################")
        
        d_dir, j_file = generate_parquet(n_rows)
        runner = Runner(n_rows, d_dir, j_file)
        
        runner.run_pandas()
        runner.run_polars(engine="cpu")
        try: runner.run_polars(engine="gpu")
        except Exception as e: print(f"  [Polars GPU] Skip: {e}")
        
        runner.run_cudf()
        runner.run_duckdb()
        
        runner.run_dask(gpu=False)
        runner.run_dask(gpu=True)
        
        runner.run_pyspark()
    
    if client: client.close()

    # ==========================================
    # 結果のCSV出力
    # ==========================================
    print("\n\nSaving results to benchmark_results.csv ...")
    if ALL_RESULTS:
        df_results = pd.DataFrame(ALL_RESULTS)
        df_results.to_csv("benchmark_results.csv", index=False)
        print(df_results)
    else:
        print("No results to save.")