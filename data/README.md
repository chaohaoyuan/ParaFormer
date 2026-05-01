# Datasets

Please download the datasets and place them in this directory.

## Medium-sized Graphs

Download from [Google Drive](https://drive.google.com/drive/folders/1rr3kewCBUvIuVxA6MJ90wzQuF-NnCRtf?usp=drive_link) and extract into this `data/` directory.

Expected structure:
```
data/
├── Planetoid/          # Cora, CiteSeer, PubMed
├── geom-gcn/           # Actor/Film, Chameleon, Squirrel, Cornell, Texas, Wisconsin
│   └── splits/         # Fixed 10-fold splits
├── wiki_new/           # Chameleon, Squirrel (new filtered splits)
├── deezer/             # Deezer-Europe
```

For Chameleon and Squirrel, we use the [new splits](https://github.com/yandex-research/heterophilous-graphs/tree/main) that filter out overlapped nodes.

## Large Graphs

- **OGB datasets** (ogbn-arxiv, ogbn-products, ogbn-proteins, ogbn-papers100M) — automatically downloaded when running the code.
- **Snap-patents, Pokec, YelpChi** — download from [Google Drive](https://drive.google.com/drive/folders/1sWIlpeT_TaZstNB5MWrXgLmh522kx4XV?usp=sharing).
