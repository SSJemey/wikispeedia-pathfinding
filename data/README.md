# Dataset

This project uses the [Wikispeedia navigation paths](https://snap.stanford.edu/data/wikispeedia.html) dataset from SNAP, Stanford. The dataset files are not included in this repository and are not covered by the project's MIT License.

The notebook downloads the data automatically. To download it manually:

1. Download `wikispeedia_paths-and-graph.tar.gz` (9.5 MB) from the link above.
2. Extract it into this folder, so that the path `data/wikispeedia_paths-and-graph/links.tsv` exists.

## Files Used

| File | Contents |
|---|---|
| `articles.tsv` | The 4,604 article names |
| `links.tsv` | Links between articles |
| `categories.tsv` | Category paths for each article |
| `paths_finished.tsv` | Finished games played by people |
| `shortest-path-distance-matrix.txt` | True shortest distances, used to check accuracy |

`paths_unfinished.tsv` is not used.

## Citations

- R. West and J. Leskovec. Human Wayfinding in Information Networks. WWW, 2012.
- R. West, J. Pineau and D. Precup. Wikispeedia: An Online Game for Inferring Semantic Distances between Concepts. IJCAI, 2009.
