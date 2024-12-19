Välkommen till min inlämning!

Här har vi tränat en model för att spela Space Invaders med hjälp av DQN

Förklaring av filer:

test_model.py - Testar en model för att se hur väl den spelar Space Invaders<br>
train.py - Skapar en ny model eller fortsätter träna en redan tränad model. Se till att ändra parametrer i config.py ifall du använder en redan tränad model.<br>
config.py - Konfiguration för modellen.<br>
utils.py - Klasser och verktyg för att få modellen att träna.<br>
requirments.txt - De biblotek som behövs för att köra koden.<br>

Andra requirments
CUDA 11.8 - Rekomenderas för GPU använding


# Setup Instructions

Follow these steps to set up the project:

## 1. Clone the Repository

```bash
git clone https://github.com/JohanNilsStje/atari_labb.git
cd atari_labb

2. Setup virtual enviorment

python -m venv .venv
.venv\Scripts\activate

3. Install requirments

pip install -r requirements.txt

4. Run the code

python train.py
