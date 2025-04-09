from transformers import pipeline
import logging

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analysis pipeline with a pre-trained model."""
        try:
            self.analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True
            )
            logging.info("Sentiment analyzer initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize sentiment analyzer: {str(e)}")
            raise
    
    def analyze(self, text):
        """
        Analyze the sentiment of the given text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: Analysis results with sentiment, score, and is_negative flag
        """
        try:
            if not text or not isinstance(text, str):
                return {
                    "sentiment": "NEUTRAL",
                    "score": 0.5,
                    "is_negative": False
                }
                
            result = self.analyzer(text[:512])[0]  # Truncate to 512 tokens
            return {
                "sentiment": result["label"],
                "score": result["score"],
                "is_negative": result["label"] == "NEGATIVE" and result["score"] > 0.7
            }
        except Exception as e:
            logging.warning(f"Sentiment analysis failed for text: {text[:50]}... Error: {str(e)}")
            return {
                "sentiment": "ERROR",
                "score": 0.0,
                "is_negative": False
            }