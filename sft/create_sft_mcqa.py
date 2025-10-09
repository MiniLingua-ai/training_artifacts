#!/usr/bin/env python3
"""
Script to create SFT (Supervised Fine-Tuning) data for multiple choice question answering.
Converts multiple choice questions from various languages into instruction-following format.
"""

import os
import random
import pandas as pd

# Set random seed for reproducibility
random.seed(42)

# Create directories for each language
languages = ["en", "es", "fr", "pt", "pl", "el", "bg", "cz", "de", "nl", "fi", "sv", "it"]

for lang in languages:
    os.makedirs(os.path.join("./sft_mqa", lang), exist_ok=True)

# Define answer format categories
letter_answers = {0, 2, 4, 7, 11, 15, 17}
number_answers = {3, 6, 8, 10, 14, 16}
full_answers = {1, 5, 9, 12, 13, 18, 19}

# English prompts
eng_prompts = [
    {
        "prompt": "Choose the correct option:\nQuestion: {your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nAnswer with the letter of the correct option.",
        "answer": "The correct answer is {answer}."
    },
    {
        "prompt": "Pick the right answer for the following:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nProvide a complete response.",
        "answer": "The correct answer is {answer}."
    },
    {
        "prompt": "Select the best response to the question below:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nRespond by selecting the letter corresponding to the correct answer.",
        "answer": "Correct option: {answer}."
    },
    {
        "prompt": "Answer the following multiple-choice question:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nRespond by selecting the number corresponding to the correct answer.",
        "answer": "The correct answer is {answer}."
    },
    {
        "prompt": "Choose the correct letter corresponding to the answer:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4}",
        "answer": "The right answer is {answer}."
    },
    {
        "prompt": "Select the right option:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nPlease give a detailed answer.",
        "answer": "Answer: {answer}."
    },
    {
        "prompt": "Choose the most appropriate option:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nProvide the number representing the correct answer.",
        "answer": "The correct answer is {answer}."
    },
    {
        "prompt": "Given the options below, select the correct answer:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nProvide the letter representing the correct answer.",
        "answer": "Correct answer: {answer}."
    },
    {
        "prompt": "Read the following and pick the best answer:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nChoose the number of the option that is correct.",
        "answer": "The correct answer is {answer}."
    },
    {
        "prompt": "Select one correct response:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nWrite out your answer in full.",
        "answer": "Right answer: {answer}."
    },
    {
        "prompt": "Choose the answer that fits best:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nPlease indicate your answer by providing the number of the correct option.",
        "answer": "The correct answer is {answer}."
    },
    {
        "prompt": "Mark the correct letter from the choices:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4}",
        "answer": "Answer: {answer}."
    },
    {
        "prompt": "Pick the option that answers the question:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nPlease give a detailed answer.",
        "answer": "Correct option is {answer}."
    },
    {
        "prompt": "Choose from the following options:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nWrite out your answer in full.",
        "answer": "The correct answer is {answer}."
    },
    {
        "prompt": "Select the answer that best completes the question:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nSelect the number that matches the correct choice.",
        "answer": "Selected answer: {answer}."
    },
    {
        "prompt": "Pick the correct one:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nSelect the letter that matches the correct choice.",
        "answer": "The correct answer is {answer}."
    },
    {
        "prompt": "Your task is to select the correct answer:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nPlease indicate your answer by providing the number of the correct option.",
        "answer": "That would be option {answer}."
    },
    {
        "prompt": "Below is a question followed by several options. Choose the right one:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nProvide the letter representing the correct answer.",
        "answer": "Answer is: {answer}."
    },
    {
        "prompt": "Review the options and select the correct one:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nProvide a complete response.",
        "answer": "The correct answer is {answer}."
    },
    {
        "prompt": "Choose the best fitting answer:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nWrite out your answer in full.",
        "answer": "Final answer: {answer}."
    }
]

# Bulgarian prompts
bg_prompts = [
    {
        "prompt": "Изберете правилния вариант:\nВъпрос: {your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nОтговорете с буквата на верния отговор.",
        "answer": "Правилният отговор е {answer}."
    },
    {
        "prompt": "Изберете верния отговор на следното:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nДайте пълен отговор.",
        "answer": "Правилният отговор е {answer}."
    },
    {
        "prompt": "Изберете най-подходящия отговор на въпроса по-долу:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nОтговорете, като изберете буквата, съответстваща на верния отговор.",
        "answer": "Верният отговор е: {answer}."
    },
    {
        "prompt": "Отговорете на следния въпрос с избор от няколко възможности:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nОтговорете, като изберете номера, съответстващ на верния отговор.",
        "answer": "Правилният отговор е {answer}."
    },
    {
        "prompt": "Изберете правилната буква, съответстваща на верния отговор:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4}",
        "answer": "Вярният отговор е {answer}."
    },
    {
        "prompt": "Изберете правилния отговор:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nМоля, дайте подробен отговор.",
        "answer": "Отговор: {answer}."
    },
    {
        "prompt": "Изберете най-подходящия вариант:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nПосочете номера, който съответства на верния отговор.",
        "answer": "Правилният отговор е {answer}."
    },
    {
        "prompt": "От предложените отговори изберете правилния:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nПосочете буквата, която съответства на верния отговор.",
        "answer": "Верният отговор е: {answer}."
    },
    {
        "prompt": "Прочетете следното и изберете най-добрия отговор:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nИзберете номера на верния отговор.",
        "answer": "Правилният отговор е {answer}."
    },
    {
        "prompt": "Изберете един верен отговор:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nНапишете отговора си изцяло.",
        "answer": "Верен отговор: {answer}."
    }
]

# Czech prompts
cz_prompts = [
    {
        "prompt": "Vyberte správnou možnost:\nOtázka: {your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nOdpovězte písmenem správné možnosti.",
        "answer": "Správná odpověď je {answer}."
    },
    {
        "prompt": "Vyberte správnou odpověď na následující otázku:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nUveďte úplnou odpověď.",
        "answer": "Správná odpověď je {answer}."
    },
    {
        "prompt": "Zvolte nejlepší odpověď na následující otázku:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nOdpovězte výběrem písmene odpovídajícího správné odpovědi.",
        "answer": "Správná volba: {answer}."
    },
    {
        "prompt": "Odpovězte na následující otázku s výběrem odpovědi:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nOdpovězte výběrem čísla odpovídajícího správné odpovědi.",
        "answer": "Správná odpověď je {answer}."
    },
    {
        "prompt": "Vyberte správné písmeno odpovídající odpovědi:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4}",
        "answer": "Správná odpověď je {answer}."
    },
    {
        "prompt": "Vyberte správnou možnost:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nProsím, uveďte podrobnou odpověď.",
        "answer": "Odpověď: {answer}."
    },
    {
        "prompt": "Zvolte nejvhodnější možnost:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nUveďte číslo, které představuje správnou odpověď.",
        "answer": "Správná odpověď je {answer}."
    },
    {
        "prompt": "Z následujících možností vyberte správnou odpověď:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nUveďte písmeno, které představuje správnou odpověď.",
        "answer": "Správná odpověď: {answer}."
    },
    {
        "prompt": "Přečtěte si následující a vyberte nejlepší odpověď:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nVyberte číslo možnosti, která je správná.",
        "answer": "Správná odpověď je {answer}."
    },
    {
        "prompt": "Zvolte jednu správnou odpověď:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nNapište svou odpověď v plném znění.",
        "answer": "Správná odpověď: {answer}."
    }
]

# German prompts
de_prompts = [
    {
        "prompt": "Wähle die richtige Option:\nFrage: {your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nAntworte mit dem Buchstaben der richtigen Option.",
        "answer": "Die richtige Antwort ist {answer}."
    },
    {
        "prompt": "Wähle die richtige Antwort für Folgendes:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nGib eine vollständige Antwort.",
        "answer": "Die richtige Antwort ist {answer}."
    },
    {
        "prompt": "Wähle die beste Antwort auf die folgende Frage:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nAntworte, indem du den Buchstaben der richtigen Antwort auswählst.",
        "answer": "Korrekte Option: {answer}."
    },
    {
        "prompt": "Beantworte die folgende Multiple-Choice-Frage:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nAntworte, indem du die Zahl der richtigen Antwort auswählst.",
        "answer": "Die richtige Antwort ist {answer}."
    },
    {
        "prompt": "Wähle den richtigen Buchstaben, der der Antwort entspricht:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4}",
        "answer": "Die richtige Antwort ist {answer}."
    },
    {
        "prompt": "Wähle die richtige Option:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nBitte gib eine ausführliche Antwort.",
        "answer": "Antwort: {answer}."
    },
    {
        "prompt": "Wähle die passendste Option:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nGib die Zahl an, die die richtige Antwort repräsentiert.",
        "answer": "Die richtige Antwort ist {answer}."
    },
    {
        "prompt": "Wähle aus den untenstehenden Optionen die richtige Antwort:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nGib den Buchstaben an, der die richtige Antwort repräsentiert.",
        "answer": "Korrekte Antwort: {answer}."
    },
    {
        "prompt": "Lies Folgendes und wähle die beste Antwort:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nWähle die Zahl der Option, die richtig ist.",
        "answer": "Die richtige Antwort ist {answer}."
    },
    {
        "prompt": "Wähle eine richtige Antwort:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nSchreibe deine Antwort vollständig aus.",
        "answer": "Richtige Antwort: {answer}."
    }
]

# Spanish prompts
es_prompts = [
    {
        "prompt": "Elige la opción correcta:\nPregunta: {your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nResponde con la letra de la opción correcta.",
        "answer": "La respuesta correcta es {answer}."
    },
    {
        "prompt": "Elige la respuesta correcta para lo siguiente:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nProporciona una respuesta completa.",
        "answer": "La respuesta correcta es {answer}."
    },
    {
        "prompt": "Selecciona la mejor respuesta a la siguiente pregunta:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nResponde seleccionando la letra correspondiente a la respuesta correcta.",
        "answer": "Opción correcta: {answer}."
    },
    {
        "prompt": "Responde la siguiente pregunta de opción múltiple:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nResponde seleccionando el número correspondiente a la respuesta correcta.",
        "answer": "La respuesta correcta es {answer}."
    },
    {
        "prompt": "Elige la letra correcta correspondiente a la respuesta:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4}",
        "answer": "La respuesta correcta es {answer}."
    },
    {
        "prompt": "Selecciona la opción correcta:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nPor favor, da una respuesta detallada.",
        "answer": "Respuesta: {answer}."
    },
    {
        "prompt": "Elige la opción más adecuada:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nProporciona el número que representa la respuesta correcta.",
        "answer": "La respuesta correcta es {answer}."
    },
    {
        "prompt": "Dadas las opciones a continuación, selecciona la respuesta correcta:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nProporciona la letra que representa la respuesta correcta.",
        "answer": "Respuesta correcta: {answer}."
    },
    {
        "prompt": "Lee lo siguiente y elige la mejor respuesta:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nElige el número de la opción que es correcta.",
        "answer": "La respuesta correcta es {answer}."
    },
    {
        "prompt": "Selecciona una respuesta correcta:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nEscribe tu respuesta completa.",
        "answer": "Respuesta correcta: {answer}."
    }
]

# French prompts
fr_prompts = [
    {
        "prompt": "Choisissez la bonne option :\nQuestion : {your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nRépondez avec la lettre de la bonne option.",
        "answer": "La bonne réponse est {answer}."
    },
    {
        "prompt": "Choisissez la bonne réponse pour ce qui suit :\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nFournissez une réponse complète.",
        "answer": "La bonne réponse est {answer}."
    },
    {
        "prompt": "Sélectionnez la meilleure réponse à la question ci-dessous :\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nRépondez en sélectionnant la lettre correspondant à la bonne réponse.",
        "answer": "Option correcte : {answer}."
    },
    {
        "prompt": "Répondez à la question à choix multiples suivante :\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nRépondez en sélectionnant le numéro correspondant à la bonne réponse.",
        "answer": "La bonne réponse est {answer}."
    },
    {
        "prompt": "Choisissez la lettre correspondant à la bonne réponse :\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4}",
        "answer": "La bonne réponse est {answer}."
    },
    {
        "prompt": "Sélectionnez la bonne option :\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nVeuillez donner une réponse détaillée.",
        "answer": "Réponse : {answer}."
    },
    {
        "prompt": "Choisissez l'option la plus appropriée :\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nIndiquez le numéro représentant la bonne réponse.",
        "answer": "La bonne réponse est {answer}."
    },
    {
        "prompt": "Parmi les options ci-dessous, sélectionnez la bonne réponse :\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nIndiquez la lettre représentant la bonne réponse.",
        "answer": "Réponse correcte : {answer}."
    },
    {
        "prompt": "Lisez ce qui suit et choisissez la meilleure réponse :\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nChoisissez le numéro de l'option correcte.",
        "answer": "La bonne réponse est {answer}."
    },
    {
        "prompt": "Sélectionnez une seule réponse correcte :\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nRédigez votre réponse en entier.",
        "answer": "Bonne réponse : {answer}."
    }
]

# Italian prompts
it_prompts = [
    {
        "prompt": "Scegli l'opzione corretta:\nDomanda: {your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nRispondi con la lettera dell'opzione corretta.",
        "answer": "La risposta corretta è {answer}."
    },
    {
        "prompt": "Scegli la risposta giusta per quanto segue:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nFornisci una risposta completa.",
        "answer": "La risposta corretta è {answer}."
    },
    {
        "prompt": "Seleziona la risposta migliore alla domanda seguente:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nRispondi selezionando la lettera corrispondente alla risposta corretta.",
        "answer": "Opzione corretta: {answer}."
    },
    {
        "prompt": "Rispondi alla seguente domanda a scelta multipla:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nRispondi selezionando il numero corrispondente alla risposta corretta.",
        "answer": "La risposta corretta è {answer}."
    },
    {
        "prompt": "Scegli la lettera corretta corrispondente alla risposta:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4}",
        "answer": "La risposta giusta è {answer}."
    },
    {
        "prompt": "Seleziona l'opzione corretta:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nPer favore, fornisci una risposta dettagliata.",
        "answer": "Risposta: {answer}."
    },
    {
        "prompt": "Scegli l'opzione più appropriata:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nIndica il numero che rappresenta la risposta corretta.",
        "answer": "La risposta corretta è {answer}."
    },
    {
        "prompt": "Dalle opzioni seguenti, seleziona la risposta corretta:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nIndica la lettera che rappresenta la risposta corretta.",
        "answer": "Risposta corretta: {answer}."
    },
    {
        "prompt": "Leggi quanto segue e scegli la risposta migliore:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nScegli il numero dell'opzione corretta.",
        "answer": "La risposta corretta è {answer}."
    },
    {
        "prompt": "Seleziona una risposta corretta:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nScrivi la tua risposta per esteso.",
        "answer": "Risposta giusta: {answer}."
    }
]

# Finnish prompts
fi_prompts = [
    {
        "prompt": "Valitse oikea vaihtoehto:\nKysymys: {your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nVastaa oikean vaihtoehdon kirjaimella.",
        "answer": "Oikea vastaus on {answer}."
    },
    {
        "prompt": "Valitse oikea vastaus seuraavaan:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nAnna täydellinen vastaus.",
        "answer": "Oikea vastaus on {answer}."
    },
    {
        "prompt": "Valitse paras vastaus alla olevaan kysymykseen:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nVastaa valitsemalla oikeaan vastaukseen liittyvä kirjain.",
        "answer": "Oikea vaihtoehto: {answer}."
    },
    {
        "prompt": "Vastaa seuraavaan monivalintakysymykseen:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nVastaa valitsemalla oikeaan vastaukseen liittyvä numero.",
        "answer": "Oikea vastaus on {answer}."
    },
    {
        "prompt": "Valitse oikea kirjain, joka vastaa oikeaa vastausta:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4}",
        "answer": "Oikea vastaus on {answer}."
    },
    {
        "prompt": "Valitse oikea vaihtoehto:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nAnna yksityiskohtainen vastaus.",
        "answer": "Vastaus: {answer}."
    },
    {
        "prompt": "Valitse sopivin vaihtoehto:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nAnna oikeaa vastausta vastaava numero.",
        "answer": "Oikea vastaus on {answer}."
    },
    {
        "prompt": "Alla olevista vaihtoehdoista valitse oikea vastaus:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nAnna oikeaa vastausta vastaava kirjain.",
        "answer": "Oikea vastaus: {answer}."
    },
    {
        "prompt": "Lue seuraava ja valitse paras vastaus:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nValitse oikean vaihtoehdon numero.",
        "answer": "Oikea vastaus on {answer}."
    },
    {
        "prompt": "Valitse yksi oikea vastaus:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nKirjoita vastauksesi kokonaisuudessaan.",
        "answer": "Oikea vastaus: {answer}."
    }
]

# Greek prompts
el_prompts = [
    {
        "prompt": "Επιλέξτε τη σωστή επιλογή:\nΕρώτηση: {your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nΑπάντησε με το γράμμα της σωστής επιλογής.",
        "answer": "Η σωστή απάντηση είναι {answer}."
    },
    {
        "prompt": "Διαλέξτε τη σωστή απάντηση για το παρακάτω:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nΔώσε μια πλήρη απάντηση.",
        "answer": "Η σωστή απάντηση είναι {answer}."
    },
    {
        "prompt": "Επιλέξτε την καλύτερη απάντηση στην παρακάτω ερώτηση:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nΑπάντησε επιλέγοντας το γράμμα που αντιστοιχεί στη σωστή απάντηση.",
        "answer": "Σωστή επιλογή: {answer}."
    },
    {
        "prompt": "Απαντήστε στην παρακάτω ερώτηση πολλαπλής επιλογής:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nΑπάντησε επιλέγοντας τον αριθμό που αντιστοιχεί στη σωστή απάντηση.",
        "answer": "Η σωστή απάντηση είναι {answer}."
    },
    {
        "prompt": "Επιλέξτε το σωστό γράμμα που αντιστοιχεί στην απάντηση:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4}",
        "answer": "Η σωστή απάντηση είναι {answer}."
    },
    {
        "prompt": "Επιλέξτε τη σωστή επιλογή:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nΠαρακαλώ δώσε μια αναλυτική απάντηση.",
        "answer": "Απάντηση: {answer}."
    },
    {
        "prompt": "Επιλέξτε την πιο κατάλληλη επιλογή:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nΔώσε τον αριθμό που αντιπροσωπεύει τη σωστή απάντηση.",
        "answer": "Η σωστή απάντηση είναι {answer}."
    },
    {
        "prompt": "Από τις παρακάτω επιλογές, επιλέξτε τη σωστή απάντηση:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nΔώσε το γράμμα που αντιπροσωπεύει τη σωστή απάντηση.",
        "answer": "Σωστή απάντηση: {answer}."
    },
    {
        "prompt": "Διαβάστε το παρακάτω και επιλέξτε την καλύτερη απάντηση:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nΕπίλεξε τον αριθμό της επιλογής που είναι σωστή.",
        "answer": "Η σωστή απάντηση είναι {answer}."
    },
    {
        "prompt": "Επιλέξτε μία σωστή απάντηση:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nΓράψε την απάντησή σου αναλυτικά.",
        "answer": "Σωστή απάντηση: {answer}."
    }
]

# Dutch prompts
nl_prompts = [
    {
        "prompt": "Kies de juiste optie:\nVraag: {your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nAntwoord met de letter van de juiste optie.",
        "answer": "Het juiste antwoord is {answer}."
    },
    {
        "prompt": "Kies het juiste antwoord voor het volgende:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nGeef een volledig antwoord.",
        "answer": "Het juiste antwoord is {answer}."
    },
    {
        "prompt": "Selecteer de beste reactie op de onderstaande vraag:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nReageer door de letter te kiezen die overeenkomt met het juiste antwoord.",
        "answer": "Juiste optie: {answer}."
    },
    {
        "prompt": "Beantwoord de volgende meerkeuzevraag:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nReageer door het nummer te kiezen dat overeenkomt met het juiste antwoord.",
        "answer": "Het juiste antwoord is {answer}."
    },
    {
        "prompt": "Kies de juiste letter die overeenkomt met het antwoord:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4}",
        "answer": "Het juiste antwoord is {answer}."
    },
    {
        "prompt": "Selecteer de juiste optie:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nGeef alstublieft een gedetailleerd antwoord.",
        "answer": "Antwoord: {answer}."
    },
    {
        "prompt": "Kies de meest geschikte optie:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nGeef het nummer dat het juiste antwoord weergeeft.",
        "answer": "Het juiste antwoord is {answer}."
    },
    {
        "prompt": "Gegeven de onderstaande opties, selecteer het juiste antwoord:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nGeef de letter die het juiste antwoord weergeeft.",
        "answer": "Juiste antwoord: {answer}."
    },
    {
        "prompt": "Lees het volgende en kies het beste antwoord:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nKies het nummer van de optie die correct is.",
        "answer": "Het juiste antwoord is {answer}."
    },
    {
        "prompt": "Selecteer één correct antwoord:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nSchrijf je antwoord volledig uit.",
        "answer": "Juiste antwoord: {answer}."
    }
]

# Polish prompts
pl_prompts = [
    {
        "prompt": "Wybierz poprawną opcję:\nPytanie: {your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nOdpowiedz, podając literę poprawnej odpowiedzi.",
        "answer": "Poprawna odpowiedź to {answer}."
    },
    {
        "prompt": "Wybierz właściwą odpowiedź na poniższe pytanie:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nUdziel pełnej odpowiedzi.",
        "answer": "Poprawna odpowiedź to {answer}."
    },
    {
        "prompt": "Wybierz najlepszą odpowiedź na poniższe pytanie:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nOdpowiedz, wybierając literę odpowiadającą poprawnej odpowiedzi.",
        "answer": "Poprawna opcja: {answer}."
    },
    {
        "prompt": "Odpowiedz na następujące pytanie wielokrotnego wyboru:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nOdpowiedz, wybierając numer odpowiadający poprawnej odpowiedzi.",
        "answer": "Poprawna odpowiedź to {answer}."
    },
    {
        "prompt": "Wybierz poprawną literę odpowiadającą odpowiedzi:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4}",
        "answer": "Właściwa odpowiedź to {answer}."
    },
    {
        "prompt": "Wybierz prawidłową opcję:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nProszę udzielić szczegółowej odpowiedzi.",
        "answer": "Odpowiedź: {answer}."
    },
    {
        "prompt": "Wybierz najbardziej odpowiednią opcję:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nPodaj numer odpowiadający poprawnej odpowiedzi.",
        "answer": "Poprawna odpowiedź to {answer}."
    },
    {
        "prompt": "Spośród poniższych opcji wybierz poprawną odpowiedź:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nPodaj literę odpowiadającą poprawnej odpowiedzi.",
        "answer": "Poprawna odpowiedź: {answer}."
    },
    {
        "prompt": "Przeczytaj poniższe i wybierz najlepszą odpowiedź:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nWybierz numer opcji, która jest poprawna.",
        "answer": "Poprawna odpowiedź to {answer}."
    },
    {
        "prompt": "Wybierz jedną poprawną odpowiedź:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nNapisz swoją odpowiedź w całości.",
        "answer": "Właściwa odpowiedź: {answer}."
    }
]

# Portuguese prompts
pt_prompts = [
    {
        "prompt": "Escolha a opção correta:\nPergunta: {your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nResponda com a letra da opção correta.",
        "answer": "A resposta correta é {answer}."
    },
    {
        "prompt": "Escolha a resposta certa para o seguinte:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nForneça uma resposta completa.",
        "answer": "A resposta correta é {answer}."
    },
    {
        "prompt": "Selecione a melhor resposta para a pergunta abaixo:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nResponda selecionando a letra correspondente à resposta correta.",
        "answer": "Opção correta: {answer}."
    },
    {
        "prompt": "Responda à seguinte pergunta de múltipla escolha:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nResponda selecionando o número correspondente à resposta correta.",
        "answer": "A resposta correta é {answer}."
    },
    {
        "prompt": "Escolha a letra correta correspondente à resposta:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4}",
        "answer": "A resposta certa é {answer}."
    },
    {
        "prompt": "Selecione a opção correta:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nPor favor, dê uma resposta detalhada.",
        "answer": "Resposta: {answer}."
    },
    {
        "prompt": "Escolha a opção mais apropriada:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nForneça o número que representa a resposta correta.",
        "answer": "A resposta correta é {answer}."
    },
    {
        "prompt": "Dadas as opções abaixo, selecione a resposta correta:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nForneça a letra que representa a resposta correta.",
        "answer": "Resposta correta: {answer}."
    },
    {
        "prompt": "Leia o seguinte e escolha a melhor resposta:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nEscolha o número da opção que está correta.",
        "answer": "A resposta correta é {answer}."
    },
    {
        "prompt": "Selecione uma resposta correta:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nEscreva sua resposta por completo.",
        "answer": "Resposta certa: {answer}."
    }
]

# Swedish prompts
sv_prompts = [
    {
        "prompt": "Välj rätt alternativ:\nFråga: {your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nSvara med bokstaven för det rätta alternativet.",
        "answer": "Rätt svar är {answer}."
    },
    {
        "prompt": "Välj rätt svar för följande:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nGe ett fullständigt svar.",
        "answer": "Rätt svar är {answer}."
    },
    {
        "prompt": "Välj det bästa svaret på frågan nedan:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nSvara genom att välja bokstaven som motsvarar det rätta svaret.",
        "answer": "Korrekt alternativ: {answer}."
    },
    {
        "prompt": "Besvara följande flervalsfråga:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nSvara genom att välja numret som motsvarar det rätta svaret.",
        "answer": "Rätt svar är {answer}."
    },
    {
        "prompt": "Välj rätt bokstav som motsvarar svaret:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4}",
        "answer": "Rätt svar är {answer}."
    },
    {
        "prompt": "Välj rätt alternativ:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nVar god och ge ett utförligt svar.",
        "answer": "Svar: {answer}."
    },
    {
        "prompt": "Välj det mest lämpliga alternativet:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nAnge numret som representerar det rätta svaret.",
        "answer": "Rätt svar är {answer}."
    },
    {
        "prompt": "Välj rätt svar bland följande alternativ:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nAnge bokstaven som representerar det rätta svaret.",
        "answer": "Korrekt svar: {answer}."
    },
    {
        "prompt": "Läs följande och välj det bästa svaret:\n{your_question_here}\n1. {option1}\n2. {option2}\n3. {option3}\n4. {option4} \n\nVälj numret på det alternativ som är rätt.",
        "answer": "Rätt svar är {answer}."
    },
    {
        "prompt": "Välj ett korrekt svarsalternativ:\n{your_question_here}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4} \n\nSkriv ut ditt svar i sin helhet.",
        "answer": "Rätt svar: {answer}."
    }
]

# Letter to number mapping
letter_to_number = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4
}

# Prompts dictionary
prompts = {
    "en": eng_prompts,
    "bg": bg_prompts,
    "cz": cz_prompts,
    "de": de_prompts,
    "es": es_prompts,
    "fr": fr_prompts,
    "it": it_prompts,
    "fi": fi_prompts,
    "el": el_prompts,
    "nl": nl_prompts,
    "pl": pl_prompts,
    "pt": pt_prompts,
    "sv": sv_prompts,
}

def main():
    """Main function to process SFT data for multiple choice questions."""
    
    # Create output directory
    os.makedirs("processed_sft", exist_ok=True)
    
    # Process each language
    for lang in os.listdir("sft_mqa"):
        if not os.path.isdir(os.path.join("sft_mqa", lang)):
            continue
            
        print(f"Processing language: {lang}")
        lang_df = pd.DataFrame(columns=["instruction", "input", "output"])
        
        # Process each file in the language directory
        for fname in os.listdir(os.path.join("sft_mqa", lang)):
            if not fname.endswith('.csv'):
                continue
                
            full_path = os.path.join("sft_mqa", lang, fname)
            print(f"  Processing file: {fname}")
            
            try:
                df = pd.read_csv(full_path)
                df = df.fillna("None")
                
                for _, row in df.iterrows():
                    # Skip rows with invalid answer keys
                    if row["answer_key"] not in ["A", "B", "C", "D"]:
                        print(f"Wrong answer key {row['answer_key']} file {full_path}")
                        continue
                    
                    # Randomly choose prompt (20% English, 80% target language)
                    value = random.random()
                    if value < 0.2 or lang not in prompts:
                        prompt_index = random.choice(list(range(len(eng_prompts))))
                        prompt = eng_prompts[prompt_index]
                    else:
                        prompt_index = random.choice(list(range(len(prompts[lang]))))
                        prompt = prompts[lang][prompt_index]
                    
                    # Format instruction
                    instruction = prompt["prompt"].format(
                        your_question_here=row["question"],
                        option1=row["A"],
                        option2=row["B"],
                        option3=row["C"],
                        option4=row["D"]
                    )
                    
                    # Format output based on prompt type
                    if prompt_index in number_answers:
                        output = prompt["answer"].format(answer=letter_to_number[row["answer_key"]])
                    elif prompt_index in letter_answers:
                        output = prompt["answer"].format(answer=row["answer_key"])
                    else:
                        output = prompt["answer"].format(answer=row[row["answer_key"]])
                    
                    # Add to dataframe
                    lang_df.loc[len(lang_df)] = [instruction, "", output]
                    
            except Exception as e:
                print(f"Error processing {full_path}: {e}")
                continue
        
        # Save processed data
        output_path = os.path.join("processed_sft", f"mcqa_{lang}.parquet")
        lang_df.to_parquet(output_path, index=False)
        print(f"Saved {len(lang_df)} samples to {output_path}")

if __name__ == "__main__":
    main()
