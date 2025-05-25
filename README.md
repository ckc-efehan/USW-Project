# 4-Tage-Woche Argument Analysis

This project analyzes arguments for and against the introduction of a 4-day work week by collecting and classifying text data from various sources.

## Project Overview

The project implements a complete data science pipeline:

1. **Data Collection**: Collects articles, posts, and tweets about the 4-day work week from:
   - News websites (Spiegel, Handelsblatt, Süddeutsche Zeitung)
   - Reddit (r/de and other relevant subreddits)
   - Twitter (with hashtags like #4TageWoche, #NewWork)

2. **Data Preparation**: Cleans and preprocesses the collected data:
   - Removes HTML tags, emojis, and irrelevant content
   - Splits texts into individual sentences
   - Creates a structured dataset for analysis

3. **Modeling**: Classifies arguments as pro, contra, or neutral using:
   - Rule-based pre-filtering with spaCy
   - Transformer-based classification with a German NLI model

4. **Evaluation**: Evaluates the model's performance:
   - Calculates accuracy and F1-scores
   - Performs error analysis
   - Visualizes results

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/4-tage-woche-analysis.git
   cd 4-tage-woche-analysis
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On Unix/MacOS
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the German language model for spaCy:
   ```
   python -m spacy download de_core_news_md
   ```

## Usage

### Running the Complete Pipeline

To run the complete pipeline with default settings:

```
python src/main.py
```

### Running Specific Steps

You can run specific steps of the pipeline using the `--steps` argument:

```
python src/main.py --steps collect,prepare
```

Available steps: `collect`, `prepare`, `model`, `evaluate`, `all`

### Data Collection

To collect data from different sources:

```
python src/main.py --steps collect --collect-news --collect-reddit --collect-twitter --search-term "4-Tage-Woche,Arbeitszeitverkürzung"
```

For Reddit and Twitter, you need to provide API credentials:

```
python src/main.py --steps collect --collect-reddit --reddit-client-id YOUR_CLIENT_ID --reddit-client-secret YOUR_CLIENT_SECRET
```

```
python src/main.py --steps collect --collect-twitter --twitter-api-key YOUR_API_KEY --twitter-api-secret YOUR_API_SECRET --twitter-access-token YOUR_ACCESS_TOKEN --twitter-access-token-secret YOUR_ACCESS_TOKEN_SECRET
```

### Data Preparation

To prepare the collected data:

```
python src/main.py --steps prepare
```

To prepare a sample for manual labeling:

```
python src/main.py --steps prepare --prepare-labeling --labeling-sample-size 100
```

### Modeling

To run the modeling component on prepared data:

```
python src/main.py --steps model
```

To use a specific model:

```
python src/main.py --steps model --model-name "svalabs/gbert-large-zeroshot-nli"
```

### Evaluation

To evaluate the model using labeled data:

```
python src/main.py --steps evaluate --labeled-file path/to/labeled_data.csv
```

## Project Structure

```
4-tage-woche-analysis/
├── data/
│   ├── raw/             # Raw data collected from various sources
│   └── processed/       # Processed and labeled data
├── src/
│   ├── data_collection/ # Scripts for collecting data
│   │   ├── news_scraper.py
│   │   ├── reddit_api.py
│   │   └── twitter_api.py
│   ├── data_preparation/ # Scripts for preprocessing data
│   │   └── data_preparation.py
│   ├── modeling/        # Scripts for modeling and classification
│   │   └── modeling.py
│   ├── evaluation/      # Scripts for evaluation
│   │   └── evaluation.py
│   └── main.py          # Main script to run the pipeline
├── requirements.txt     # Required packages
└── README.md            # This file
```

## Data Sources

- **News Websites**: Spiegel Online, Handelsblatt, Süddeutsche Zeitung
- **Reddit**: r/de, r/arbeitsleben, r/Finanzen
- **Twitter**: Tweets with hashtags like #4TageWoche, #NewWork

## Models

- **Rule-based Classification**: Uses spaCy for pattern matching and keyword detection
- **Transformer-based Classification**: Uses the `svalabs/gbert-large-zeroshot-nli` model from Huggingface for zero-shot classification

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The project was developed as part of a data science course
- Thanks to all the open-source libraries and models used in this project