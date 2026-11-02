"""Test métamorphique d'invariance : une faute de frappe ne doit pas changer le sentiment.

Relation : sentiment(phrase) == sentiment(faute_de_frappe(phrase)).
On ne vérifie jamais que le sentiment est « juste », seulement qu'il est cohérent.

Usage :
    pip install transformers torch pytest
    pytest -q -rf test_faute_de_frappe.py
Le nombre de « failed » dans le résumé est le nombre de phrases (sur 40) dont le
sentiment change à cause de la faute.
"""
import pytest
from transformers import pipeline

OPINIONS = [
    "the service was excellent",
    "the food was terrible",
    "the movie was a waste of time",
    "the hotel was wonderful",
    "the support team was really helpful",
    "the delivery was awful",
    "the show was boring",
    "the new phone is fantastic",
    "the course was disappointing",
    "the restaurant was great",
]

CADRES = [
    "John said that @OP@.",
    "According to John, @OP@.",
    "John wrote in a review that @OP@.",
    "I asked John and the answer was that @OP@.",
]

PHRASES = [cadre.replace("@OP@", op) for cadre in CADRES for op in OPINIONS]


def faute_de_frappe(phrase: str) -> str:
    """Inverse les 2e et 3e lettres du mot le plus long."""
    mots = phrase.split()
    i = max(range(len(mots)), key=lambda k: len(mots[k]))
    m = mots[i]
    mots[i] = m[0] + m[2] + m[1] + m[3:]
    return " ".join(mots)


@pytest.fixture(scope="session")
def modele():
    clf = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )
    return lambda texte: clf(texte)[0]["label"]


@pytest.mark.parametrize("phrase", PHRASES)
def test_faute_de_frappe(modele, phrase):
    avant = modele(phrase)
    apres = modele(faute_de_frappe(phrase))
    # même sentiment attendu
    assert apres == avant
