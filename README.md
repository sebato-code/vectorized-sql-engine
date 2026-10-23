# Vectorized SQL Engine

Lightweight, vectorized embedded relational database engine in Python with custom Buffer Pool Manager and Disk B+ Tree index.

## Features
- **Paging:** Fixed-size 4096-byte disk pages with DiskManager.
- **Buffer Pool:** LRU/Clock eviction policy with dirty page management.
- **Index:** On-disk B+ Tree for logarithmic lookup and fast sequential range scans.
- **Vectorized Engine:** Volcano iterator execution by tuple batches.
- **Operators:** SeqScan, Filter, HashJoin, and Aggregates.
