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
        #self.products = [
        #   "Mortgage", 
        #    "Credit Card", 
        #    "Savings Account",
        #    "Personal Loan",
        #    "Business Banking",
        #    "Investment Account"
        #]
        self.products = [
            "GPP", 
            "GSP", 
            "GSIPP", 
            "SWMT"
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
        """Generate realistic pension product customer feedback text."""
        positive = [
            "Very happy with my GPP pension transfer process",
            "The GSIPP online portal is excellent for managing my investments",
            "Great advice from my pension advisor about my SWMT options",
            "GSP performance has exceeded my expectations this year",
            "The pension calculator helped me understand my GPP projections",
            "Clear communication about my GSIPP fees and charges",
            "Easy to make additional contributions to my SWMT online"
        ]
        
        negative = [
            "Extremely disappointed with GPP customer service response times",
            "My GSP transfer took weeks to complete - unacceptable!",
            "GSIPP portal keeps crashing when I try to view statements",
            "SWMT performance has been terrible this quarter",
            "No one could explain the charges on my GPP statement",
            "GSP annual statement contained multiple errors",
            "GSIPP withdrawal process is far too complicated"
        ]
        
        neutral = [
            "Received my annual GPP statement as expected",
            "The SWMT online portal is functional but could be improved",
            "Standard service for my GSP query, nothing exceptional",
            "GSIPP performance is average compared to other providers",
            "Had to call twice to get my pension question answered",
            "The pension calculator gives reasonable estimates",
            "Annual review meeting was satisfactory but not outstanding"
        ]
        
        return random.choice(positive + negative + neutral)
    
    def _generate_resolution_notes(self):
        """Generate realistic resolution notes for pension product feedback."""
        resolutions = [
            "Explained the GPP transfer process timeline in detail",
            "Arranged pension specialist callback to discuss GSP performance",
            "Waived admin fee for GSIPP as goodwill gesture",
            "Provided detailed breakdown of SWMT investment charges",
            "Escalated GPP portal issues to technical team",
            "Offered free pension review session to address concerns",
            "Corrected GSP statement errors and sent revised copy",
            "Explained tax implications of GSIPP withdrawals"
        ]
        return random.choice(resolutions)
