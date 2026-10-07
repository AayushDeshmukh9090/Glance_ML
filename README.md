# Triple-Stream Fashion Search Engine

A fashion image retrieval system that searches images using three types of information:

- **Fashion facts** such as clothing attributes and colors
- **Context and style** from image captions
- **Visual features** from the image itself

The three streams are combined at search time so the system can handle simple, contextual, and compositional fashion queries.

> **Demo video:** [View the Web UI demo](https://drive.google.com/file/d/1OnA6LFqLni2XDpjmIwVKiXBlMu7tDq45/view?usp=sharing)

## Setup

### Requirements

- Python 3.8+
- 8 GB+ RAM
- 5 GB+ free disk space
- CUDA-capable GPU recommended for BLIP-2

### 1. Clone the repository

```bash
git clone https://github.com/AayushDeshmukh9090/Glance_ML.git
cd Glance-ML
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Install the Fashionpedia API package:

```bash
cd fashionpedia-api-master
pip install -e .
cd ..
```

Main libraries used:

- `chromadb` - vector database
- `transformers` - CLIP and BLIP-2
- `torch` - deep learning
- `streamlit` - web interface
- `Pillow` - image processing
- `pyyaml` - configuration

## Use the Pre-indexed Database

The easiest way to run the project is to use the pre-built database.

Download:

```text
chroma_db/
logs/
outputs/
```

from the Google Drive link above and place them in the project root.

You can then start the application with:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Build the Database from Scratch

### 1. Download Fashionpedia

Place the dataset inside `data/`.

Expected files include:

```text
instances_attributes_train2020.json
attributes_train2020.json
info_test2020.json
train/
```

The dataset contains 45,623 fashion images.

Update the paths in:

```text
shared/config.yaml
```

when required.

### 2. Run the indexing pipeline

```bash
bash bash_files/indexer_pipeline.sh
```

The pipeline:

1. Extracts Fashionpedia attributes and colors
2. Generates BLIP-2 captions
3. Encodes images with CLIP
4. Stores the resulting vectors in ChromaDB

The full indexing process takes several hours depending on the hardware. The pipeline supports checkpointing and can resume from saved progress.

## Project Structure

```text
Glance-ML/
├── app.py
├── indexer/
│   ├── indexer.py
│   ├── caption_generator.py
│   ├── README.md
│   └── __init__.py
├── retriever/
│   ├── retriever.py
│   ├── evaluate.py
│   ├── optimize_weights.py
│   ├── README.md
│   └── __init__.py
├── shared/
│   ├── utils.py
│   ├── logger.py
│   ├── config.yaml
│   └── __init__.py
├── data/
├── chroma_db/
├── logs/
├── outputs/
├── indexer_pipeline.sh
├── retriever_pipeline.sh
├── requirements.txt
└── README.md
```

## Architecture

Each image is represented using three separate vectors:

| Stream | Source | Purpose |
|---|---|---|
| **V_fact** | Fashionpedia attributes + colors | Structured fashion information |
| **V_vibe** | BLIP-2 captions | Scene, style, and occasion |
| **V_img** | CLIP image encoder | Visual information |

At query time, the streams are combined using:

```text
Score = α·S_fact + β·S_vibe + γ·S_img
```

This allows the system to give more importance to the stream that matches the query.

For example:

- Attribute queries can emphasize fashion attributes
- Style queries can emphasize contextual descriptions
- Visual queries can rely more on image features

## Key Features

- Triple-stream vector search
- Dynamic query-time weighting
- Fashion attribute and color extraction
- BLIP-2 based scene and style captions
- CLIP-based image and text embeddings
- Fashion-specific query expansion
- Compositional queries such as `"red tie + white shirt"`
- Configurable weight presets
- Batch retrieval and latency tracking

## Search Examples

The system supports different types of queries:

1. **Attribute**
   > A person in a bright yellow raincoat

2. **Context**
   > Professional business attire inside a modern office

3. **Semantic**
   > Someone wearing a blue shirt sitting on a park bench

4. **Style**
   > Casual weekend outfit for a city walk

5. **Compositional**
   > A red tie and a white shirt in a formal setting

## Running Retrieval

Run the retriever directly:

```bash
python retriever/retriever.py
```

Or use:

```bash
./retriever_pipeline.sh
```

Logs are stored in:

```text
logs/retriever.log
```

Typical query latency after model warm-up is around **30–40 ms** on the measured setup.

## Query-Time Weighting

Instead of merging all embeddings into one vector during indexing, the system keeps the three streams separate.

This makes it possible to change the weights depending on the query:

```text
Score = α·S_fact + β·S_vibe + γ·S_img
```

Example optimized weights:

| Query type | α | β | γ |
|---|---:|---:|---:|
| Attribute-specific | 0.008 | 0.660 | 0.331 |
| Contextual place | 0.217 | 0.266 | 0.517 |
| Complex semantic | 0.467 | 0.237 | 0.296 |
| Style inference | 0.446 | 0.514 | 0.040 |
| Compositional | 0.038 | 0.568 | 0.394 |

These weights were optimized using Bayesian optimization with Optuna.

## Evaluation

The triple-stream system was compared with a vanilla CLIP baseline using Precision@10.

| Query type | Triple-Stream | Vanilla CLIP | Improvement |
|---|---:|---:|---:|
| Compositional | 100% | 80% | +25% |
| Attribute-specific | 40% | 10% | +300% |
| Complex semantic | 10% | 0% | +100% |
| **Average** | **50%** | **30%** | **+66.7%** |

Evaluation can be run with:

```bash
python retriever/evaluate.py
```

The relevance check uses keyword matching against metadata, so the automatic scores may not fully capture semantic relevance for abstract style or context queries.

## Performance

The indexed dataset contains:

- **45,623 images**
- **136,869 vectors**
- 3 vector collections

Measured indexing time on the DGX setup:

| Stage | Time |
|---|---:|
| Grounded layer generation | 249 min |
| BLIP-2 caption generation | 199 min |
| Vector encoding + ChromaDB | 17 min |
| **Total** | **465 min (~7.75 hours)** |

Query performance:

- First query: ~234 ms
- Subsequent queries: ~30–40 ms

The project uses batching, checkpointing, and batched ChromaDB inserts to improve processing speed and memory usage.

## Technical Details

### Color Extraction

- K-means clustering with `k=3`
- Uses segmentation masks
- Maps RGB values to 20 fashion color names
- Uses `"neutral"` as a fallback for very small masks

### Scene and Style Captions

BLIP-2 is used to generate context-aware captions from images.

Example:

```text
A person wearing a red wool blazer and blue slim-fit jeans
standing in a modern office environment
```

### Score Normalization

ChromaDB returns distances where lower values indicate closer matches.

The default normalization is:

```text
1 - (distance / max_distance)
```

Other supported methods include exponential and inverse normalization.

## Configuration

The main configuration file is:

```text
shared/config.yaml
```

It contains weight presets and query-expansion rules.

Example:

```yaml
weight_presets:
  attribute_specific:
    alpha: 0.4
    beta: 0.1
    gamma: 0.5

  style_inference:
    alpha: 0.2
    beta: 0.7
    gamma: 0.1

expansion_rules:
  weekend: "relaxed leisure street casual"
  bright yellow: "yellow vibrant sunny golden"
  red: "crimson scarlet burgundy"
```

## Reproduce the Evaluation

```bash
./run_evaluation.sh
```

Detailed results are saved in:

```text
evaluation_results.json
```

The evaluation uses:

- Precision@10
- Keyword-based relevance matching
- Triple-stream weighted fusion
- Vanilla CLIP as the baseline

## Limitations

- The color taxonomy is limited to 20 basic colors
- BLIP-2 can struggle with unclear or ambiguous scenes
- Some semantically related clothing terms may not always retrieve the expected results
- Automatic keyword-based evaluation does not capture all semantic matches

## Future Work

Possible improvements include:

- Cross-modal attention for stream fusion
- Fine-tuning CLIP on Fashionpedia
- Hard-negative mining for compositional queries
- Real-time context using weather or location data
- User feedback and click-through optimization

## Author

**Aayush Deshmukh**

## License

This project uses the Fashionpedia dataset. See:

```text
fashionpedia-api-master/license.txt
```

for the applicable dataset license.

## Acknowledgments

- Fashionpedia
- OpenAI CLIP
- Salesforce BLIP-2
- ChromaDB
