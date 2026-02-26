# Traffic-Predictor

## Overview
Traffic-Predictor is a project that analyzes and predicts traffic patterns using the US Accidents dataset. The dataset contains information about accidents in the United States, including location, time, and contributing factors.

## Dataset
The dataset used for this project is sourced from Kaggle:
[US Accidents Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

## Project Structure
```
.gitignore
app.py                # Main application script
eda.py                # Exploratory Data Analysis script
mlmodel.py            # Machine Learning model script
README.md             # Project documentation
requirements.txt      # Python dependencies
data/                 # Contains raw and cleaned datasets
```

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/Traffic-Predictor.git
   cd Traffic-Predictor
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

## Contributing

A traffic event is of one of the
following types: accident, broken-vehicle2
, congestion3
, construction4
,
event5
, lane-blocked6
, or flow-incident7

## License
This project is licensed under the MIT License.

## Acknowledgments
- Dataset: [US Accidents Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

- Regular Expression Patterns. Given the description of traffic
events of type accident, we were able to identify 27 regular expression patterns; 16 of them were extracted based on MapQuest
data, and 11 from Bing data. Among the MapQuest patterns, the
following expression corresponds to junctions (see Table 1): “. . . on
. . . at exit . . .”, and the following pattern mostly10 determines an
intersection: “. . . on . . . at . . .”. We consider a location an intersection if it is associated with at-least one of the following annotations
(see Table 1): crossing, stop, or traffic signal. Among Bing regular expression patterns, two of them identify junctions: “at . . . exit . . .”
and “ramp to . . .”. Table 2 shows several examples of accidents
