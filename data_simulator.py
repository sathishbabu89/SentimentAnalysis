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
        
        # UK cities with coordinates
        self.uk_locations = {
            # England
            "London": {"lat": 51.5074, "lon": -0.1278, "country": "England"},
            "Manchester": {"lat": 53.4808, "lon": -2.2426, "country": "England"},
            "Birmingham": {"lat": 52.4862, "lon": -1.8904, "country": "England"},
            "Leeds": {"lat": 53.8008, "lon": -1.5491, "country": "England"},
            "Liverpool": {"lat": 53.4084, "lon": -2.9916, "country": "England"},
            "Bristol": {"lat": 51.4545, "lon": -2.5879, "country": "England"},
            "Sheffield": {"lat": 53.3811, "lon": -1.4701, "country": "England"},
            "Newcastle": {"lat": 54.9783, "lon": -1.6178, "country": "England"},
            # Scotland
            "Edinburgh": {"lat": 55.9533, "lon": -3.1883, "country": "Scotland"},
            "Glasgow": {"lat": 55.8642, "lon": -4.2518, "country": "Scotland"},
            # Wales
            "Cardiff": {"lat": 51.4837, "lon": -3.1681, "country": "Wales"},
            "Swansea": {"lat": 51.6214, "lon": -3.9436, "country": "Wales"},
            # Northern Ireland
            "Belfast": {"lat": 54.5973, "lon": -5.9301, "country": "Northern Ireland"}
        }
        
        # Banking products
        self.products = [
            "Current Account", 
            "Mortgage", 
            "Credit Card", 
            "Savings Account",
            "Personal Loan",
            "Business Banking",
            "Investment Account"
        ]
        
        # Customer service staff
        self.staff_members = [
            "John Smith", "Emma Johnson", "Michael Brown", 
            "Sarah Davis", "David Wilson", "Lisa Miller",
            "Robert Taylor", "Jennifer Anderson", "William Thomas"
        ]

    def generate_sample_data(self, num_records=100):
        """
        Generate synthetic customer feedback data with sentiment analysis.
        
        Args:
            num_records (int): Number of records to generate
            
        Returns:
            pd.DataFrame: Generated data with sentiment analysis
        """
        channels = ["Twitter", "Facebook", "Live Chat", "App Review", "Call Transcript", "Email"]
        
        data = []
        for _ in range(num_records):
            # Select a random UK location
            location = random.choice(list(self.uk_locations.keys()))
            location_data = self.uk_locations[location]
            
            record = {
                "timestamp": datetime.now() - timedelta(minutes=random.randint(0, 1440)),
                "channel": random.choice(channels),
                "region": location,
                "country": location_data["country"],
                "lat": location_data["lat"],
                "lon": location_data["lon"],
                "product": random.choice(self.products),
                "customer_id": self.fake.uuid4(),
                "staff_member": random.choice(self.staff_members),
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
            "Quick response to my query via the chat feature. Excellent!",
            "The branch staff were extremely helpful with my account setup.",
            "Interest rates on my savings account are very competitive."
        ]
        
        negative = [
            "Extremely frustrated with the mortgage application process!",
            "App keeps crashing when I try to make payments. Unacceptable!",
            "Waited 45 minutes to speak to someone. Service is deteriorating.",
            "Incorrect charges on my account and no one can explain why.",
            "Website keeps logging me out mid-transaction. Very frustrating!",
            "My card was blocked without warning causing huge inconvenience.",
            "Mortgage advisor gave me incorrect information about rates."
        ]
        
        neutral = [
            "Opened a new savings account. Process was straightforward.",
            "Received my new debit card in the mail as expected.",
            "The interest rates seem competitive compared to other banks.",
            "Had to visit branch to complete the application as online didn't work.",
            "Standard service, nothing exceptional but no complaints either.",
            "The mobile app is okay but could use some improvements.",
            "Average experience, neither good nor bad."
        ]
        
        return random.choice(positive + negative + neutral)
    
    def _generate_resolution_notes(self):
        """Generate realistic resolution notes for negative feedback."""
        resolutions = [
            "Apologized to customer and offered £20 goodwill gesture",
            "Issue escalated to technical team for investigation",
            "Provided detailed explanation of charges and customer satisfied",
            "Arranged callback from specialist team within 24 hours",
            "Walked customer through app troubleshooting steps",
            "Replaced customer's card and waived replacement fee",
            "Offered rate review and improved customer's mortgage terms",
            "Explained security procedures that led to card block"
        ]
        return random.choice(resolutions)
