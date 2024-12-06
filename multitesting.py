from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import sigmoid
import matplotlib.pyplot as plt

# Load pretrained model and tokenizer
model_name = "monologg/bert-base-cased-goemotions-original"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# GoEmotions class labels (replace this with the full list of GoEmotions labels if available)
class_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", 
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", 
    "fear", "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse", 
    "sadness", "surprise", "neutral"
]

def predict_emotions(texts):
    """
    Predict emotions for a list of texts using the GoEmotions model.

    Args:
        texts (list): List of input texts.
        threshold (float): Threshold for multi-class classification.

    Returns:
        list: List of dictionaries containing input text, predicted emotions, and probabilities.
    """
    # Tokenize and encode the input texts
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    
    # Get model outputs
    outputs = model(**inputs)
    logits = outputs.logits
    probs = sigmoid(logits)  # Convert logits to probabilities
    
    # Process predictions for each input
    results = []
    for i, text in enumerate(texts):
        # Get the top 2 probabilities and their indices
        top_probs, top_indices = torch.topk(probs[i], k=4)
        predicted_emotions = [idx for idx in top_indices]
        
        # Append the results
        results.append({
            "text": text,
            "predicted_emotions": predicted_emotions,
            "probabilities": top_probs.detach().numpy()
        })
    return results

if __name__ == "__main__":
    # Example input texts
    input_texts = [
    "You know the answer man, you are programmed to capture those codes they send you, don’t avoid them!",
    "I've never been this sad in my life!",
    "The economy is heavily controlled and subsidized by the government. In any case, I was poking at the lack of nuance in US politics today.",
    "He could have easily taken a real camera from a legitimate source and change the price in Word/Photoshop and then print it out.",
    "Thank you for your vote of confidence, but we statistically can't get to 10 wins.",
    "Wah Mum other people call me on my bullshit and I can't ban them, go outside son.",
    "There it is!",
    "At least now [NAME] has more time to gain his confidence.",
    "Good. We don't want more thrash liberal offspring in this world.",
    "It's better to say a moment like that could truly ignite her love for the game rather than putting a bit of a damper on it.",
    "I went to a destination wedding being the only single person. Promised to never put myself in that situation again.",
    "He died 4 days later of dehydration.",
    "Like this just cuz of the [NAME] rhymes background raps...but dude your [NAME] is sick against [NAME].",
    "Lol dream on buddy. You’ve had enough attention today. Actually learn what your talking about helps a lot. Sorry your stuck in free roam smokin crack.",
    "As an anesthesia resident this made me blow air out my nose at an accelerated rate for several seconds. Take your damn upvote you bastard.",
    "1-2-3-4 I declare a thumb war! Dangit [NAME], you win again! Ok you get to stab me again :(",
    "Did you hear the reason for this? Because they are concerned about inventory, initially.",
    "[NAME] is such a legendary daddy 😩.",
    "I don't necessarily hate them, but then again, I dislike it when people breed while knowing how harsh life is.",
    "Hoarders unite! 2388 stars here, 19 rewards, 20 by tomorrow. Highest reward was 23. No Visa card. Too tired of playing credit card roulette.",
    "Downvoted to hell but I understand your experience. Salute, soldier.",
    "You aren't a real 90's kid if you were born after 1999."
    ]


    
    # Predict emotions
    threshold = 0.5
    predictions = predict_emotions(input_texts)
    
    # Display predictions
    for result in predictions:
        print(f"Text: {result['text']}")
        print(f"Predicted Emotions: {result['predicted_emotions']}")
        print(f"Probabilities: {result['probabilities']}\n")

