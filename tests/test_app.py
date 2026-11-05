from pathlib import Path

import pytest

# ces tests tournent dans le job "app" de la CI, installé comme sur Streamlit Cloud (seulement app/requirements.txt)
# dans le job principal streamlit n'est pas installé, ils sont donc sautés
testing = pytest.importorskip("streamlit.testing.v1")

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_starts():
    # l'app démarre, télécharge le modèle depuis la release et affiche sa réponse sans erreur
    at = testing.AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, [e.message for e in at.exception]
    assert len(at.slider) == 1 and len(at.selectbox) == 2


def test_guarantee_slider():
    # en changeant la garantie, l'app recalcule sans planter
    at = testing.AppTest.from_file(APP, default_timeout=120).run()
    at.slider[0].set_value(99).run()
    assert not at.exception, [e.message for e in at.exception]
