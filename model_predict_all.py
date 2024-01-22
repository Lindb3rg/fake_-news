from model_predict import preprocess_text, loaded_logistic_model,loaded_sequential_model,\
    loaded_svm_model,loaded_tfidf_vectorizer,loaded_tokenizer

from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode



def model_predict_text(original_text, modelname="svm"):
    preprocessed_texts = [preprocess_text(x) for x in original_text]

    maxlen = 75

    if modelname == "svm" or modelname == "logistic":
        processed_text_strings = [' '.join(text) for text in preprocessed_texts]
        tfidf_features = loaded_tfidf_vectorizer.transform(processed_text_strings)
        return tfidf_features

    elif modelname == "sequential":
        new_sequences = loaded_tokenizer.texts_to_sequences(preprocessed_texts)
        new_padded_sequence = pad_sequences(new_sequences, maxlen=maxlen)
        return new_padded_sequence

    else:
        print("Error in preprocessing")

def vote_setup(texts, modelnames, models):
    
    model_name = None
    list_of_preprocessed_texts = []
    for index, model in enumerate(models):
        if modelnames[index] == "svm" or modelnames[index] == "logistic" and model_name == None:
            base_model_preprocessed_texts = model_predict_text(texts, modelnames[index])
            list_of_preprocessed_texts.append(("base_model",base_model_preprocessed_texts))
            model_name = modelnames[index]
        elif modelnames[index] == "sequential":
            sequential_model_preprocessed_texts = model_predict_text(texts, modelnames[index])
            list_of_preprocessed_texts.append(("sequential_model",sequential_model_preprocessed_texts))

    list_of_predictions = []

    for index, preprocessed_text in enumerate(list_of_preprocessed_texts):
        if preprocessed_text[0] == "base_model":
            svm_predictions = loaded_svm_model.predict(preprocessed_text[1])
            svm_predictions = svm_predictions.reshape(-1, 1)
            list_of_predictions.append(svm_predictions)
            logistic_predictions = loaded_logistic_model.predict(preprocessed_text[1])
            logistic_predictions = logistic_predictions.reshape(-1, 1)
            list_of_predictions.append(logistic_predictions)
        elif preprocessed_text[0] == "sequential_model":
            keras_predictions = loaded_sequential_model.predict(preprocessed_text[1])
            keras_predictions = keras_predictions.reshape(-1, 1)
            keras_predictions_binary = (keras_predictions > 0.5).astype(int)
            list_of_predictions.append(keras_predictions_binary)

    all_predictions = np.hstack([x for x in list_of_predictions])
    voting_predictions, _ = mode(all_predictions, axis=1)


    # Print the majority prediction and corresponding model names for each instance
    print("Majority Predictions:")
    for i, majority_prediction in enumerate(voting_predictions):
        # Find all models with the majority prediction for this instance
        majority_models_indices = np.where(all_predictions[i] == majority_prediction)[0]

        # Filter indices that are within the valid range of model_names
        valid_indices = [idx for idx in majority_models_indices if 0 <= idx < len(modelnames)]

        if len(valid_indices) > 0:
            majority_models = [modelnames[idx] for idx in valid_indices]
            print(f"Instance {i + 1}: Majority Prediction - {majority_prediction} (by {', '.join(majority_models)})")
        else:
            print(f"Instance {i + 1}: No models predicted the majority class for this instance")



def bar_plot_all_predictions(all_predictions, predictions_models, modelnames):

    all_predictions = np.hstack(predictions_models)

    gold_color = (1.0, 0.84, 0.0)
    dark_blue_color = (0.0, 0.0, 0.5)

    # Count the occurrences of each prediction for each model
    counts_0 = np.sum(all_predictions == 0, axis=0)
    counts_1 = np.sum(all_predictions == 1, axis=0)
    total_instances = len(all_predictions)

    # Calculate percentages
    percentage_0 = (counts_0 / total_instances) * 100
    percentage_1 = (counts_1 / total_instances) * 100

    # Set the figure size and style
    # sns.set(style="whitegrid")  # Use a seaborn style
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting the grouped bar chart
    width = 0.35  # Width of each bar
    ind = np.arange(len(modelnames))

    bars_0 = ax.bar(ind - width/2, percentage_0, width, label='Predicted 0', color=gold_color)
    bars_1 = ax.bar(ind + width/2, percentage_1, width, label='Predicted 1', color=dark_blue_color)

    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Percentage', fontsize=14)
    ax.set_title('Percentage of Predictions for True and Fake by All Models', fontsize=16)
    ax.set_xticks(ind)
    ax.set_xticklabels(modelnames, fontsize=12)
    ax.legend(fontsize=12)

    # Add percentage labels on top of each bar with a different color
    for bar_0, bar_1 in zip(bars_0, bars_1):
        height_0 = bar_0.get_height()
        height_1 = bar_1.get_height()
        
        ax.annotate(f'{height_0:.2f}%', xy=(bar_0.get_x() + bar_0.get_width() / 2, height_0),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color='black', fontsize=10)
        
        ax.annotate(f'{height_1:.2f}%', xy=(bar_1.get_x() + bar_1.get_width() / 2, height_1),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color='black', fontsize=10)

    # Set the y-axis scale to show 0-100%
    ax.set_yticks(np.arange(0, 101, 5))
    # Add grid lines
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.1, color=dark_blue_color, which='both')

    plt.show()


def distribution_majority_predictions(voting_predictions):
    unique_majority_predictions, counts = np.unique(voting_predictions, return_counts=True)

    # Plotting the bar chart
    plt.bar(unique_majority_predictions, counts, color='skyblue')
    plt.xlabel('Majority Prediction')
    plt.ylabel('Number of Instances')
    plt.title('Distribution of Majority Predictions')
    plt.xticks(unique_majority_predictions)
    plt.show()
