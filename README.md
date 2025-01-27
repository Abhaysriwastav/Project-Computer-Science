# Automated Mental Health Monitoring Using AI

## Introduction

This project develops an innovative automated mental health monitoring system powered by artificial intelligence. In response to the growing prevalence of mental health issues and barriers to accessible care, this system aims to provide timely, personalized support through advanced natural language processing and machine learning techniques.

## Project Objectives

1. Address the stigma surrounding mental health discussions
2. Offer proactive, AI-driven mental health support
3. Increase accessibility to mental health resources
4. Reduce dependence on traditional care methods that may be unavailable or inaccessible
5. Provide real-time analysis and interventions based on user interactions

## Methodology
Our system leverages cutting-edge AI technologies to analyze user data and provide personalized mental health insights:

1. **Natural Language Processing (NLP)**: Processes and analyzes text inputs from users, including responses to questionnaires and potentially social media posts.
2. **Machine Learning (ML)**: Employs supervised learning algorithms to classify text into emotional categories and detect indicators of mental health conditions.
3. **Deep Learning**: Utilizes neural networks, specifically leveraging TensorFlow and TensorFlow Hub, for advanced text classification and sentiment analysis.
4. **Data Preprocessing**: Implements robust data cleaning and transformation techniques to prepare text data for analysis.
5. **Sentiment Analysis**: Categorizes text into binary emotional states (positive/negative) to gauge overall mental well-being.

## System Architecture

1. **Data Collection Module**: Gathers user inputs through interactive questionnaires and potentially social media integrations.
2. **Preprocessing Pipeline**: Cleans and prepares text data for analysis, including tokenization and normalization.
3. **ML Model**: A sequential neural network built with TensorFlow, featuring:
   - An embedding layer from TensorFlow Hub for text vectorization
   - Dense layers for classification
   - Binary cross-entropy loss for optimization
4. **Real-time Analysis Engine**: Processes user inputs and generates immediate mental health scores.
5. **Feedback System**: Provides personalized advice and recommendations based on the analysis results.

## Implementation Details
1. Data Collection and Preprocessing:

The first step in our system is collecting and preparing the data. This process is crucial because the quality of our data directly impacts the performance of our AI model.
Our main data collection happens through the `preprocess(file)` function. Here's how it works:

```python
def preprocess(file):
    path = os.getcwd() + file + '.txt'
    data = pd.read_csv(path, sep = ';')
    hos = []
    for i in range(len(data.emotion)):
        if data['emotion'][i] in ['joy', 'love', 'surprise']:
            hos.append(1)  # happy is 1
        else:
            hos.append(0)  # sad is 0
    data['hos'] = hos
    return data
```

This function does several important things:

a) It reads a CSV file containing our training or validation data. The file is expected to have two columns: 'text' and 'emotion'.
b) It then creates a new binary classification called 'hos' (happy or sad). This is a simplification of the original emotion categories, which helps our model focus on the overall sentiment rather than specific emotions.
c) The function considers 'joy', 'love', and 'surprise' as positive emotions (labeled as 1), while all other emotions are considered negative (labeled as 0).

This preprocessing step is crucial because it transforms our multi-class emotion classification problem into a binary sentiment analysis problem, which is often easier for machine learning models to handle, especially with limited data.

2. Model Architecture and Training:
Our model is built using TensorFlow and TensorFlow Hub. Here's how we define and compile our model:

```python
model = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(model, output_shape=[20], input_shape=[], 
                           dtype=tf.string, trainable=True)

model = tf_keras.Sequential()
model.add(hub_layer)
model.add(tf_keras.layers.Dense(16, activation='relu'))
model.add(tf_keras.layers.Dense(1))

model.compile(optimizer='adam',
              loss=tf_keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf_keras.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])
```

Let's break this down:

a) We start by using a pre-trained text embedding layer from TensorFlow Hub. This layer converts our text inputs into 20-dimensional vectors, capturing semantic meanings of words.
b) We then add a dense layer with 16 units and ReLU activation. This layer helps our model learn complex patterns in the embedded text.
c) The final layer is a single unit dense layer, which outputs a single value. This value represents the likelihood of the input text being associated with a positive sentiment.
d) We compile the model using the Adam optimizer and binary cross-entropy loss, which are standard choices for binary classification problems.

The model is then trained using the `fit()` method:

```python
history = model.fit(train.text,
                    train.hos,
                    epochs=40,
                    batch_size=512,
                    validation_data=(val.text, val.hos),
                    verbose = 0)
```

This trains the model for 40 epochs, using a batch size of 512. The model's performance is evaluated on the validation set after each epoch.

3. Prediction and Post-processing:

After training, we use the model to make predictions:

```python
predstr = model.predict(train.text)
```

However, the raw predictions from our model aren't very interpretable. That's where our `postprocessor(preds)` function comes in:

```python
def postprocessor(preds):
    range = predstr.max() - predstr.min()
    norm_preds = []
    probab = []
    for i in preds:
        norm_preds.append((i - predstr.min()) / range)
        probab.append((i - predstr.min()) * 100 / range)
    return np.mean(probab)
```

This function does a few important things:

a) It normalizes our predictions to fall between 0 and 1.
b) It then converts these normalized predictions into percentages (0-100).
c) Finally, it returns the mean of these percentages.

This post-processing step is crucial because it transforms our model's raw outputs into a more understandable "mental health score" between 0 and 100.

4. User Interaction and Evaluation:

The `evaluation_start()` function is where our system interacts with the user:

```python
def evaluation_start():
    answers = []
    questions = [
        'How would you describe your experience at your workplace/college/school in the past few days? \n',
        'How do you like to spend your leisure time? How do you feel after it?\n',
        'Life has its ups and downs. Although handling successes can be difficult, setbacks can affect mental health strongly. How do you manage your emotions after failures? \n',
        'Are there any improvements/decline in your salary/grades? \n',
        'Any recent unpleasant experience that you would like to share?\n',
        'In a broad sense, how would you describe the way your life is going on?\n',
        'How would you describe your experience at your workplace/college/school in the past few days?\n'
    ]
    
    for question in questions:
        answers.append(input(question))
    
    results = model.predict(answers)
    score = postprocessor(results)
    print('Your mental health score is:', score)
    
    if score < 25:
        print("You are going through a bad phase in life. But don't worry, bad times are not permanent. Try to seek help from a trained professional to improve your mental health.")
    else:
        print("Your mental health looks great! Continue enjoying life and try to help others who are struggling with their mental health.")
```

This function does several things:

a) It presents a series of questions to the user, designed to gauge their current mental state.
b) It collects the user's responses and feeds them into our trained model.
c) It then uses the `postprocessor()` function to convert the model's predictions into a single mental health score.
d) Finally, it provides feedback to the user based on their score, offering encouragement and suggesting professional help if the score is low.

This function serves as the main interface between our AI system and the user, translating the complex workings of our model into actionable insights and advice.
In conclusion, our Automated Mental Health Monitoring system is a complex interplay of data preprocessing, machine learning, and user interaction. Each component plays a crucial role in transforming raw text inputs into meaningful mental health insights. The system demonstrates how AI can be leveraged to provide accessible, continuous mental health support, potentially serving as a valuable complement to traditional mental health care methods.
Remember, while this system can provide helpful insights, it's always important to seek professional help for serious mental health concerns. This AI system is designed to be a supportive tool, not a replacement for professional mental health care.


### Main Components

- `Automated Health Monitoring System.py`: Core Python script containing:
  - Data preprocessing functions
  - Model definition and training procedures
  - Evaluation function with interactive questionnaire

- `train.txt`: Dataset used for training the model (format: text;emotion)
- `val.txt`: Validation dataset for model evaluation

### Key Functions

1. `preprocess(file)`: Prepares data by loading CSV and binarizing emotions.
2. `postprocessor(preds)`: Normalizes model predictions into a 0-100 scale.
3. `evaluation_start()`: Initiates the interactive questionnaire and provides mental health scoring.

### Model Architecture

```python
model = tf_keras.Sequential([
    hub_layer,
    tf_keras.layers.Dense(16, activation='relu'),
    tf_keras.layers.Dense(1)
])
```

The model uses a pre-trained text embedding layer from TensorFlow Hub, followed by dense layers for classification.

## Usage Guide

1. Ensure all dependencies are installed (TensorFlow, Pandas, NumPy, TensorFlow Hub).
2. Run the `Automated Health Monitoring System.py` script.
3. Follow the prompts to answer the mental health questionnaire.
4. Receive a mental health score and personalized advice based on your responses.

## Future Enhancements

1. Integration with social media platforms for broader data collection
2. Implementation of more sophisticated NLP techniques for nuanced emotion detection
3. Development of a user-friendly mobile application or web interface
4. Incorporation of additional mental health metrics and assessment tools
5. Collaboration with mental health professionals to refine advice and recommendations

## Ethical Considerations

This system is designed as a supportive tool and should not replace professional mental health care. Users are encouraged to seek professional help for serious mental health concerns. The project prioritizes user privacy and data security in all aspects of its implementation.

## Contributors

- Abhay Sriwastav

## Acknowledgments

We extend our gratitude to the open-source community, particularly the TensorFlow and TensorFlow Hub teams, for providing the tools and resources that made this project possible.
