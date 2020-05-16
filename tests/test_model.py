#!/usr/bin/env python
import pytest
from disaster_tweets.models.model import Model

def test_predict():
    model = Model()
    model.predict([
        ["Theyd probably still show more life than Arsenal did yesterday, eh? EH?"],
        ["Just happened a terrible car crash"]
    ])
