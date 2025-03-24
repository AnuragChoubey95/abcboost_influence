#!/bin/zsh

# Usage: ./run_abcboost.zsh <dataset_substring>
# Example: ./run_abcboost.zsh compas

# 1) Check argument
if [ $# -lt 1 ]; then
  echo "Usage: $0 <dataset_substring>"
  exit 1
fi

SUBSTR=$1
ZSH_MAP_FILE="dataset_task_map.sh"

# 2) Extract task from dataset_task_map.sh
TASK=$(
  grep "\\[$SUBSTR\\]=" "$ZSH_MAP_FILE" |
  sed -E 's/^.*="([^"]*)".*/\1/' |
  sed -E 's/^[[:space:]]+|[[:space:]]+$//g'
)
if [ -z "$TASK" ]; then
  echo "Could not find valid entry for '$SUBSTR' in $ZSH_MAP_FILE."
  exit 1
fi

echo "Found dataset '$SUBSTR' with task type '$TASK'."

# 3) Build train/test file paths
TRAIN_FILE="data/${SUBSTR}.train.csv"
TEST_FILE="data/${SUBSTR}.test.csv"

# Make sure they exist
if [ ! -f "$TRAIN_FILE" ]; then
  echo "Train file '$TRAIN_FILE' not found!"
  exit 1
fi
if [ ! -f "$TEST_FILE" ]; then
  echo "Test file '$TEST_FILE' not found!"
  exit 1
fi

# 4) Translate $TASK into an abcboost method (use arrays so each token is distinct)
LP_FLAG=()
EXTRA_FLAGS=()
MODEL_SUFFIX=""

if [ "$TASK" = "binary" ]; then
  METHOD="robustlogit"
  # no extra flags
elif [ "$TASK" = "multiclass" ]; then
  METHOD="mart"
  EXTRA_FLAGS=("-search" "2" "-gap" "10")
elif [ "$TASK" = "regression" ]; then
  METHOD="regression"
  LP_FLAG=("-lp" "2")
  MODEL_SUFFIX="_p2"
else
  echo "Unknown task type: $TASK"
  exit 1
fi

echo "Training method: $METHOD"

# 5) Construct the abcboost_train command
TRAIN_CMD=(
  "./abcboost_train"
  "-method" "$METHOD"
  "-data" "$TRAIN_FILE"
  "-J" "20"
  "-v" "0.1"
  "-iter" "1000"
)

# If regression, add ("-lp" "2"); if multiclass, add ("-search" "2" "-gap" "10")
TRAIN_CMD+=("${LP_FLAG[@]}")
TRAIN_CMD+=("${EXTRA_FLAGS[@]}")

# 6) Construct model name, e.g.
# - For binary: compas.train.csv_robustlogit_J20_v0.1.model
# - For regression: concrete.train.csv_regression_J20_v0.1_p2.model
MODEL_NAME="${SUBSTR}.train.csv_${METHOD}_J20_v0.1${MODEL_SUFFIX}.model"
# If you want to actually save the model, uncomment:
# TRAIN_CMD+=("-modelsave" "$MODEL_NAME")

echo "Running training command: ${TRAIN_CMD[@]}"
"${TRAIN_CMD[@]}"

# 7) Construct predict command
PRED_CMD=(
  "./abcboost_predict"
  "-data" "$TEST_FILE"
  "-model" "$MODEL_NAME"
)

# Output a prediction file named, e.g. compas_robustlogit_J20_v0.1.prediction or
# compas_regression_J20_v0.1_p2.prediction if regression. 
PRED_FILE="${SUBSTR}_${METHOD}_J20_v0.1${MODEL_SUFFIX}.prediction"
# If you want to actually save predictions, uncomment:
# PRED_CMD+=("-predict" "$PRED_FILE")

echo "Running prediction command: ${PRED_CMD[@]}"
"${PRED_CMD[@]}"

echo "Done. Model stored as $MODEL_NAME, predictions as $PRED_FILE."
