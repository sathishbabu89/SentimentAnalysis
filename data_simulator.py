import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
from sentiment_analyzer import SentimentAnalyzer
import logging

class DataSimulator:
    def __init__(self):
        """Initialize the data simulator with Faker and SentimentAnalyzer."""
        self.analyzer = SentimentAnalyzer()
        self.fake = Faker('en_UK')
        logging.basicConfig(level=logging.INFO)
        
    def generate_sample_data(self, num_records=100):
        """
        Generate synthetic customer feedback data with sentiment analysis.
        
        Args:
            num_records (int): Number of records to generate
            
        Returns:
            pd.DataFrame: Generated data with sentiment analysis
        """
        channels = ["Twitter", "Facebook", "Live Chat", "App Review", "Call Transcript"]
        regions = ["London", "Manchester", "Birmingham", "Leeds", "Edinburgh", "Cardiff"]
        products = ["Current Account", "Mortgage", "Credit Card", "Savings", "Loan"]
        staff_members = ["John Smith", "Emma Johnson", "Michael Brown", 
                        "Sarah Davis", "David Wilson", "Lisa Miller"]
        
        data = []
        for _ in range(num_records):
            record = {
                "timestamp": datetime.now() - timedelta(minutes=random.randint(0, 1440)),
                "channel": random.choice(channels),
                "region": random.choice(regions),
                "product": random.choice(products),
                "customer_id": self.fake.uuid4(),
                "staff_member": random.choice(staff_members),
                "feedback_text": self._generate_feedback_text(),
                "resolution_time": random.randint(1, 120) if random.random() > 0.7 else None
            }
            
            # Add sentiment analysis
            sentiment = self.analyzer.analyze(record["feedback_text"])
            record.update(sentiment)
            
            # Add simulated resolution data if negative
            if record["is_negative"] and random.random() > 0.5:
                record["resolution_status"] = random.choice(["Resolved", "Pending", "Escalated"])
                record["resolution_notes"] = self._generate_resolution_notes()
            else:
                record["resolution_status"] = None
                record["resolution_notes"] = None
                
            data.append(record)
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def _generate_feedback_text(self):
        """Generate realistic banking customer feedback text."""
        positive = [
            "Great service from Lloyds today! The mobile app works perfectly.",
            "Very helpful customer service representative solved my issue quickly.",
            "The new online banking features are fantastic and easy to use.",
            "I'm very satisfied with my mortgage advisor's guidance.",
            "Quick response to my query via the chat feature. Excellent!"
        ]
        
        negative = [
            "Extremely frustrated with the mortgage application process!",
            "App keeps crashing when I try to make payments. Unacceptable!",
            "Waited 45 minutes to speak to someone. Service is deteriorating.",
            "Incorrect charges on my account and no one can explain why.",
            "Website keeps logging me out mid-transaction. Very frustrating!"
        ]
        
        neutral = [
            "Opened a new savings account. Process was straightforward.",
            "Received my new debit card in the mail as expected.",
            "The interest rates seem competitive compared to other banks.",
            "Had to visit branch to complete the application as online didn't work.",
            "Standard service, nothing exceptional but no complaints either."
        ]
        
        return random.choice(positive + negative + neutral)
    
    def _generate_resolution_notes(self):
        """Generate realistic resolution notes for negative feedback."""
        resolutions = [
            "Apologized to customer and offered £20 goodwill gesture",
            "Issue escalated to technical team for investigation",
            "Provided detailed explanation of charges and customer satisfied",
            "Arranged callback from specialist team within 24 hours",
            "Walked customer through app troubleshooting steps"
        ]
        return random.choice(resolutions)