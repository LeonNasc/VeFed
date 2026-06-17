"""
PhraseLibrary — medium-fidelity symptom text generator.

Sits between raw templates (zero variance) and Ollama (full LLM inference).
Hand-written phrases per (disease × severity × personality) bucket, sampled
at runtime. Rich enough in lexical diversity for DistilBERT to learn both
disease and severity signals without any network calls.

Usage
-----
    lib = PhraseLibrary(seed=42)
    record = lib.sample("influenza", "moderate", personality="neutral", days=3)
    # returns a dict ready to feed to FLLearner: {text, label, gt_disease, gt_severity}
"""
from __future__ import annotations
import random
from typing import Optional


# ── Phrase library ─────────────────────────────────────────────────────────────
# Keys: PHRASES[disease][severity][personality] = list[str]
# {days} placeholder is filled at sample time.
# Personality keys: "stoic" | "neutral" | "anxious"
# Disease keys:     "influenza" | "pneumonia" | "non_infectious"
# Severity keys:    "discharge" | "mild" | "moderate" | "severe" | "critical"

PHRASES: dict[str, dict[str, dict[str, list[str]]]] = {

    # ══════════════════════════════════════════════════════════════════════════
    # INFLUENZA
    # Hallmarks: sudden onset, systemic (myalgia, bone-deep ache, headache),
    # fever+chills, dry cough, fatigue, sore throat. Chest not dominant.
    # ══════════════════════════════════════════════════════════════════════════
    "influenza": {

        "discharge": {
            "stoic": [
                "I had some aches and a bit of a temperature a few days ago but it's mostly gone now. Felt fine to come in.",
                "I think I'm over the worst of it — had some muscle aches and a mild fever but they're fading. Probably didn't even need to come.",
                "Been feeling a bit off with headaches and fatigue for {days} day(s), seems to be clearing up.",
                "Had the usual flu stuff — achy, feverish — for about {days} day(s) but it's resolving on its own.",
                "I'm almost back to normal. Had body aches and chills but nothing dramatic. Just checking in.",
                "It started as a bad headache and muscle soreness. {days} day(s) later and I'm nearly fine.",
                "I won't pretend it was fun — had a proper fever and aches — but I'm over it now. Just wanted it documented.",
                "Bit of a rough patch, mainly aches and a temperature, but honestly I'm feeling much better.",
                "The fever broke yesterday. I still have a bit of tiredness but the body aches are gone.",
                "Had chills and felt really run down for a couple of days. I think it was just a short flu.",
                "I had a temperature of about 38 and some joint pain for {days} day(s). Coming down now.",
                "The worst seems to be behind me — had shivering, aches, and fatigue but the fever is gone.",
                "I've been a bit rough for {days} day(s) with a temperature and muscle soreness but I can function normally again.",
                "I had a sudden fever and felt drained for a day or two. It's going away without treatment.",
                "Flu symptoms — achy, tired, feverish — for {days} day(s) but they're resolving.",
            ],
            "neutral": [
                "I had flu-like symptoms approximately {days} day(s) ago — fever peaking at 38.2, myalgia, headache. Resolving now.",
                "Temperature was elevated for {days} day(s), along with generalised muscle aches and fatigue. Improving without treatment.",
                "Sudden onset {days} day(s) ago: fever 38.5, body aches, mild sore throat, dry cough. Symptoms now subsiding.",
                "I experienced typical influenza symptoms — chills, sweating, headache, myalgia — for {days} day(s). Currently recovering.",
                "Fever reached 38.8 on day one. Muscle pain, fatigue, mild headache. Now on day {days} and symptoms are nearly gone.",
                "Onset was abrupt: within hours I had fever, aches, and exhaustion. {days} day(s) in and it is resolving.",
                "Influenza-type illness: sudden fever, profound myalgia, dry cough. Duration {days} day(s), improving.",
                "Self-limited febrile illness with generalised aches and fatigue. Lasted {days} day(s), now nearly resolved.",
                "Body temperature measured 38.4 at home. Concurrent myalgia, headache, rhinorrhoea. Recovering over {days} day(s).",
                "I had systemic symptoms consistent with influenza for {days} day(s). Feeling substantially better today.",
                "Fever, chills, and bone aches for {days} day(s). Appetite is returning, temperature back to normal.",
                "Classic flu picture — sudden onset, high fever, severe tiredness — for {days} day(s). Recovering well.",
                "Myalgia and fever lasting {days} day(s). No chest symptoms. Improving without intervention.",
                "I had a temperature spike and significant fatigue {days} day(s) ago. Both are resolving.",
                "Brief febrile illness with aches and headache, lasted {days} day(s), currently near resolution.",
            ],
            "anxious": [
                "I've had fever and body aches for {days} day(s) and I just wanted to make absolutely sure I'm fully in the clear.",
                "I seem to be getting better but I'm still worried — I had such bad muscle pain and a high temperature, I want someone to confirm I'm okay.",
                "It felt like the flu — chills, aching everywhere — and although it's been improving I want to be checked out.",
                "I had a 38.5 fever and terrible aches for {days} day(s). I think I'm recovering but I'm keeping a close eye on it.",
                "The fever seems to be going but I'm still nervous after feeling so awful with the aches and exhaustion.",
                "I was really unwell — headache, body pain, fever — and I want to make sure there are no complications.",
                "I measured my temperature multiple times. It got to 38.9 on day one. Today it's 37.4 but I still don't feel 100%.",
                "After {days} day(s) of flu symptoms I'm better but I'd feel much better having a doctor confirm everything is normal.",
                "My muscles were aching so badly I could barely move. The fever's gone but I'm checking in just to be safe.",
                "I had every flu symptom I could think of and I just want to be reassured that it's completely resolved.",
                "Even though I feel better today, after {days} day(s) of high fever and intense aches I want a professional opinion.",
                "I've been tracking my temperature religiously — it went from 38.7 down to 37.0 over {days} day(s). Am I really clear?",
                "I had chills so bad I was shaking — it was quite frightening — but it seems to be passing. Should I be worried about anything?",
                "The aches and fever are mostly gone but I keep checking if I have a temperature. I just want confirmation I'm fine.",
                "I'm probably overreacting but after feeling that unwell with the flu symptoms I wanted a professional opinion before going back to work.",
            ],
        },

        "mild": {
            "stoic": [
                "I've had a bit of a temperature and some aches for {days} day(s), nothing I can't handle.",
                "Probably just a mild flu. A bit achy, bit of a headache, slight fever. I'm managing fine.",
                "I don't want to make a fuss but I've had muscle soreness and a low fever for {days} day(s).",
                "Feeling slightly under the weather — headache, some tiredness, mild fever — for about {days} day(s).",
                "I've had a low-grade temperature and generalised aches for {days} day(s). It's not stopping me functioning.",
                "Just a touch of flu I think. Chills and aches but nothing serious.",
                "I've been a little off — tired, mildly feverish, some muscle pain. {days} day(s) now.",
                "It started with a bit of a sore throat and some aching. Now there's a mild fever. {days} day(s) in.",
                "I'm a bit run down with aches and a temperature but I wouldn't call it serious.",
                "Had better days, that's all. Mild fever, some joint aches, nothing alarming.",
                "The aching is the main thing — my arms and legs feel heavy. There's a mild fever too. {days} day(s).",
                "I've been mildly feverish and achy. Probably just a touch of flu. I'd rather not be here but my partner insisted.",
                "A day or two of low-grade fever and muscle soreness. Could be worse.",
                "I feel like I'm coming down with something — mild temperature, aching all over, bit of a dry cough.",
                "Not great but not terrible. Headache, chills, fatigue. About {days} day(s) of this.",
            ],
            "neutral": [
                "Low-grade fever around 37.8–38.2°C for {days} day(s). Generalised myalgia, mild headache, some fatigue.",
                "I have mild influenza symptoms — body aches, low fever, sore throat — for {days} day(s). No dyspnoea.",
                "Onset {days} day(s) ago with mild fever, dry cough, and muscle pain. Able to carry out daily activities.",
                "Generalised aches and a temperature of 38.0. Going on {days} day(s). No significant respiratory symptoms.",
                "Mild systemic illness: fever 37.9, myalgia, headache. Duration {days} day(s). No chest involvement.",
                "I have a low fever and muscle soreness. Came on fairly suddenly {days} day(s) ago.",
                "Temperature is 38.1. Aches mainly in legs and back. Some fatigue. Started {days} day(s) ago.",
                "Classic flu-like illness at mild severity — fever, chills, myalgia — for {days} day(s).",
                "I've been measuring my temperature: consistently 37.8–38.0. Muscle aches and fatigue alongside it.",
                "Mild febrile illness with systemic aches and headache. Duration {days} day(s). Able to rest at home.",
                "Low-grade fever, dry cough, and mild myalgia. No significant respiratory distress. {days} day(s) duration.",
                "I feel mildly unwell with aches and a temperature but no chest symptoms or severe fatigue.",
                "Fever 38°C, generalised soreness, some chills. Started suddenly {days} day(s) ago.",
                "Body aches, headache, mild fever. I can still move around but feel run down. {days} day(s).",
                "I have a mild flu. Systemic symptoms — fever, aches — but no alarm features.",
            ],
            "anxious": [
                "I've had a fever and body aches for {days} day(s) and I keep worrying it might develop into something worse.",
                "I'm not sure if this is just a mild flu or something more serious — I have a temperature and my whole body aches.",
                "I've been achy and feverish for {days} day(s) and I'm worried because I've read about complications.",
                "My temperature is 38.1 and I have significant muscle pain. I want to get it checked before it gets worse.",
                "I have a mild fever and aches but I'm quite anxious about it — it came on so suddenly.",
                "I measured 38.3 this morning. I have aches and feel exhausted. Should I be worried at all?",
                "The headache and body pain have been with me for {days} day(s). I feel like I should be here just in case.",
                "I'm probably overreacting but with a fever and aching all over I want to make sure it's nothing serious.",
                "I've been tracking my temperature every few hours — it's hovering around 38. The aches are quite bad.",
                "I feel feverish and my muscles hurt. I know it might just be flu but I want it confirmed.",
                "I have a mild temperature and muscle soreness. Not severe, but I'm keeping a close eye on it.",
                "I'm feeling genuinely worried — the body aches came on so suddenly along with the fever.",
                "I have mild flu symptoms but my immune system isn't the best and I'd rather have someone look at me.",
                "Body aches and a low fever for {days} day(s). I've been monitoring closely because I tend to go downhill fast.",
                "I feel mildly unwell but I'm the kind of person who likes to catch things early. Fever, aches, headache.",
            ],
        },

        "moderate": {
            "stoic": [
                "I've been fairly unwell for {days} day(s) — high fever, muscle pain — but I'm managing at home.",
                "Pretty rough past {days} day(s). Body aches all over, headache, fever. Couldn't go to work but I'm coping.",
                "I've had a proper flu I think. Took a couple of days off work. Fever, aches, very tired.",
                "It's been unpleasant. High temperature, aching joints, some chills. {days} day(s) of it.",
                "I'm not at my best. Fever's been up and down, aching everywhere, headache behind my eyes.",
                "Significant flu symptoms for {days} day(s). Mostly just resting. I'll live.",
                "High fever and bad muscle pain. I've been in bed for most of it. It's not nothing.",
                "I had to call in sick. Body aches that are hard to describe — feels deep in the bones. Fever over 39.",
                "I've been sweating and shivering on and off for {days} day(s) and the aching is significant.",
                "The headache has been the worst part, behind my eyes and temples. Plus a temperature over 38.5.",
                "I've been horizontal for most of {days} day(s). Fever, aches, fatigue. Clearly influenza.",
                "Pretty unwell but I've had worse. Fever around 39, muscle soreness, dry cough. {days} day(s).",
                "It's been a difficult week. I'd say moderate flu — I'm not hospitalised but I'm not functioning normally.",
                "Fever spiking to 39.2, full-body aches, sweats. I've been off work for {days} day(s).",
                "I feel genuinely ill — not emergency-room ill — but I've been confined to bed with fever and aches.",
            ],
            "neutral": [
                "Fever ranging 38.5–39.2°C for {days} day(s). Significant myalgia, pronounced headache, intermittent chills and sweats.",
                "High fever, generalised myalgia, fatigue preventing normal activity. Onset {days} day(s) ago, sudden.",
                "Temperature 39°C, severe body aches, headache localised retro-orbitally. Dry cough. No dyspnoea.",
                "Influenza-compatible illness: abrupt onset {days} day(s) ago, fever 39.1, profound myalgia, fatigue.",
                "I've been febrile at 38.7–39.3 for {days} day(s) with significant muscle and joint pain. Reduced to bed rest.",
                "Moderate-to-severe myalgia and fever. Measured 39.0 this morning. Headache and sweats overnight.",
                "Flu syndrome — fever, chills, sweating, aches — at moderate severity for {days} day(s). No chest pain.",
                "Temperature has been 38.8–39.4. Body aches preventing normal daily activity. {days} day(s) duration.",
                "I have a high fever and significant systemic symptoms. Headache, myalgia, fatigue. Onset was sudden.",
                "Febrile illness 39°C, myalgia, headache, dry cough. Reduced daily function for {days} day(s).",
                "Fever peaked at 39.5 on day one. Persistent aches and fatigue. Now on day {days}, still significant symptoms.",
                "Flu-like illness: sudden onset, high fever, body aches, chills. Moderately severe for {days} day(s).",
                "I've been febrile with temperature 38.9–39.2 and severe myalgia for {days} day(s).",
                "Pronounced systemic symptoms: high fever, rigors, myalgia, frontal headache. Duration {days} day(s).",
                "I have influenza. Temperature 39°C, body aches, I cannot go to work. Started {days} day(s) ago.",
            ],
            "anxious": [
                "I feel terrible and I'm not getting better. High fever for {days} day(s), aching everywhere, I can't sleep properly.",
                "I'm really worried — my temperature has been over 39 for {days} day(s) and the body aches are unbearable.",
                "This has been going on {days} day(s) and the fever isn't breaking. I'm scared it's going to get worse.",
                "The muscle pain is the worst I've ever had and the fever keeps spiking. I need someone to look at me.",
                "I've been in bed for {days} day(s). Fever, chills, sweating — I cycle through all of them. I'm very worried.",
                "My temperature hit 39.4 last night. The body aches are severe and I'm worried about complications.",
                "I can't function. Fever won't break, body aches are awful, I'm exhausted. Please tell me this is normal.",
                "I've been tracking my temperature obsessively — it went from 39.1 to 38.6 but I'm still in a lot of pain.",
                "The headache and aching have been relentless for {days} day(s). I'm frightened it's not just flu.",
                "I'm not improving the way I expected. High fever, severe muscle pain — should I be in hospital?",
                "My whole body hurts and I've had a fever over 39 for {days} day(s). I keep thinking it might be something more serious.",
                "I'm exhausted and in significant pain. Temperature is 38.8, aches are everywhere. I feel like I'm deteriorating.",
                "I've been shivering and sweating alternately for {days} day(s). Is this normal for flu?",
                "The fever and aches started so suddenly I thought something was really wrong. {days} day(s) later and I'm still very unwell.",
                "I want to be seen urgently — {days} day(s) of high fever and I don't feel like I'm improving.",
            ],
        },

        "severe": {
            "stoic": [
                "I've been bedbound for {days} day(s). Fever spiking to 40, severe muscle pain, barely able to get up.",
                "I won't pretend it's mild — I've been very unwell. Fever over 39.5, unable to walk without help.",
                "This is serious flu. I haven't been able to eat or drink properly for {days} day(s) and the fever is very high.",
                "I've been essentially immobile with fever and aches for {days} day(s). Someone had to drive me here.",
                "It's bad. Temperature 40, the aches are severe, I've been in bed solidly. I don't get ill like this normally.",
                "I'm not someone who comes to the doctor but this is not right. Fever hitting 40, body pain that's stopped me moving.",
                "Very unwell for {days} day(s). Fever won't break, I can't keep fluids down, the aching is severe.",
                "I've had flu before but this is different in intensity — very high fever, extreme fatigue, muscle pain.",
                "I can't function at all. Fever 39.8, rigors, severe myalgia. I needed help getting here.",
                "Someone had to help me come in today. Fever over 40, aching is severe, I'm quite dehydrated.",
                "I've been feverish at 40°C for {days} day(s) and the body pain is preventing me from doing anything.",
                "This is the worst I've felt in years. Severe flu symptoms — very high fever, total fatigue, intense aching.",
                "I've had {days} day(s) of severe flu. High fever, significant muscle pain, I cannot eat.",
                "Lying flat with a temperature near 40 and terrible aches. I needed to be brought in by family.",
                "The fever is very high and the body aches are severe. I'm quite unwell and I need proper assessment.",
            ],
            "neutral": [
                "Fever 39.8–40.2°C for {days} day(s). Severe myalgia, unable to mobilise without assistance. Reduced oral intake.",
                "High fever sustained at 40°C, profound myalgia, significant fatigue. Duration {days} day(s). Dehydration risk.",
                "Severe influenza presentation: temperature 40°C, rigors, bone-deep muscle pain, bed-bound for {days} day(s).",
                "I have been febrile at 39.5–40°C for {days} day(s) with severe generalised aches and extreme fatigue.",
                "Temperature 40°C, unable to eat or drink adequately, severe myalgia. Brought in by family after {days} day(s).",
                "Fever peaked at 40.3. Severe systemic symptoms — aching, weakness, rigors. Now on day {days}.",
                "I've been unable to leave bed for {days} day(s). Fever 39.9, severe muscle and joint pain.",
                "Influenza at high severity. Fever 40°C, profound fatigue, severe myalgia. Family assisted attendance.",
                "Temperature 40.1 sustained. Severe body aches, minimal fluid intake, rigors at night. {days} day(s) duration.",
                "Serious influenza: high fever, very severe myalgia, unable to carry out self-care for {days} day(s).",
                "Fever 39.8, severe whole-body aches, I am barely ambulatory. Duration {days} day(s).",
                "I have a temperature of 40 and severe muscle pain. I've been horizontal for {days} day(s).",
                "High-severity flu illness: sustained fever >39.5, profound systemic aches, requiring assistance.",
                "Severe febrile illness with temperature 40°C and extreme myalgia. Day {days}.",
                "Sustained fever 39.7–40.2°C, severe muscular pain, dehydration developing. Duration {days} day(s).",
            ],
            "anxious": [
                "I'm really frightened — my temperature has been at 40 for {days} day(s) and I can barely move.",
                "I think I need urgent help. Extremely high fever, severe body pain, I'm not able to function at all.",
                "This is the most ill I've ever been. Fever 40, aches that are making me cry. I'm very worried.",
                "I've been in agony with body aches and a very high temperature for {days} day(s). I couldn't delay coming in.",
                "Please help me — my temperature hit 40.2 last night and the muscle pain is unbearable.",
                "I'm scared. I can't eat, can't drink, my temperature is at 40 and the aches are the worst I've ever had.",
                "I've been deteriorating for {days} day(s). The fever is very high, I'm severely achy, I can't care for myself.",
                "I was brought here by my family — I've been too unwell to come alone. Fever over 40, severe pain.",
                "My temperature keeps going higher and the body pain is extreme. I don't know how much worse this can get.",
                "I'm terrified this is going to develop into something worse — very high fever, severe aching, {days} day(s) now.",
                "I need someone to tell me I'm going to be okay — my fever is 40, I'm in severe pain, I can't function.",
                "This illness has completely floored me. Fever 39.9, aches beyond anything I've experienced. Very worried.",
                "I can barely speak through the exhaustion and pain. Fever has been high for {days} day(s). I'm frightened.",
                "I've been tracking my temperature every two hours — it hasn't gone below 39.5. The muscle pain is severe.",
                "Please assess me carefully — I feel very unwell with a very high fever and severe aches. I'm really concerned.",
            ],
        },

        "critical": {
            "stoic": [
                "I'm in a bad way. Fever over 40, I can barely stay upright. I needed to come in.",
                "I don't want to alarm anyone but I can't stand without help, my temperature is 40.5, and I'm confused.",
                "Things have escalated. Very high fever, I'm not entirely with it, I needed someone to help me get here.",
                "I've had severe flu that's getting worse. Extremely high temperature, I'm not myself.",
                "I'm unwell in a way I can't manage at home. Very high fever, severe pain, impaired thinking.",
                "This is beyond what I can handle at home. Temperature above 40, disoriented, unable to stand alone.",
                "I'll be brief — I feel very seriously ill. Very high fever, I'm not thinking straight, severe body pain.",
                "I'm not coping. My temperature is 40.6, I'm dizzy and confused, I couldn't come without help.",
                "I've been severely unwell for {days} day(s) — extremely high fever, I've become confused, I needed an ambulance.",
                "I don't normally come to the doctor but I genuinely couldn't manage at home. Very high fever, severe pain, altered.",
            ],
            "neutral": [
                "Temperature 40.5–41°C. Severe confusion, unable to stand without assistance. Acute febrile illness, day {days}.",
                "Critical influenza presentation: fever 40.8°C, acute confusion, profound weakness, requiring ambulance transport.",
                "Extremely high fever >40.5°C, acute delirium, severe myalgia, cannot self-care. Duration {days} day(s).",
                "Fever 40.6, confusion onset today. Severe systemic illness. Brought in by family from bed.",
                "Acute deterioration: fever now 41°C, altered consciousness, extreme myalgia. Emergency presentation.",
                "High-severity influenza with confusion: temperature 40.7, disoriented, severely weak. Day {days}.",
                "Fever 40.5°C with acute encephalopathic features — confusion, inability to walk. Emergency. Day {days}.",
                "Critical febrile illness: sustained fever >40°C, acute confusion, complete inability to self-care.",
                "Very high fever, severe confusion, cannot stand without help. Brought by ambulance on day {days}.",
                "Extreme myalgia and very high fever with new confusion. Emergency admission from home.",
            ],
            "anxious": [
                "Please help me — I'm burning up, I can't think straight, my fever is over 40 and I'm terrified.",
                "I'm very frightened. My temperature is 40.5, I'm confused, I nearly fell getting up. I came by ambulance.",
                "Something is very wrong. High fever, I'm not thinking properly, I'm in so much pain. I'm extremely worried.",
                "I feel like I'm dying — extremely high fever, I'm confused, I can barely move. Please help me.",
                "I've never been this ill. Fever above 40, I lost track of time, I couldn't get here alone. I'm scared.",
                "My family called an ambulance for me. I have a very high fever, I'm confused, I'm in severe pain.",
                "I don't know what's happening to me — very high fever, I've been confused since this morning. Please help.",
                "This is a crisis. Temperature 40.6, I'm not right in my head, I can barely stand. I'm terrified.",
                "I'm frightened to my core. High fever, confusion, severe pain — I've been deteriorating all day.",
                "Please take this seriously — fever at 40.5, I'm disoriented, I need immediate help.",
            ],
        },
    },  # end influenza

    # ══════════════════════════════════════════════════════════════════════════
    # BACTERIAL PNEUMONIA
    # Hallmarks: productive cough with colored sputum, chest pain (often
    # pleuritic — worse on breathing), shortness of breath, fever, less
    # myalgia than flu. More insidious or subacute onset in many cases.
    # ══════════════════════════════════════════════════════════════════════════
    "pneumonia": {

        "discharge": {
            "stoic": [
                "I had a chesty cough with some yellow mucus for a few days, it's mostly cleared up now.",
                "I've been getting over a chest infection. The cough is nearly gone and my breathing is back to normal.",
                "Had a productive cough and some chest discomfort for {days} day(s). It's resolving.",
                "I went through a rough patch with a deep cough and fever. I think I'm nearly better.",
                "The cough with green phlegm has almost stopped and my chest feels clearer. Just a check-in.",
                "I had some pleuritic-type chest pain and a productive cough but it's resolved mostly.",
                "My breathing is back to normal and the cough is drying up. It was {days} day(s) of it.",
                "Had a chest infection — cough, fever, difficulty breathing — but I'm on the mend.",
                "The productive cough was worrying me but it's improved significantly over the past day or two.",
                "I had shortness of breath and a chesty cough for {days} day(s). I can breathe normally again now.",
                "My chest feels much clearer. The deep cough and colored phlegm are mostly gone.",
                "I've been through {days} day(s) of a chest infection. Coughing up phlegm, chest discomfort. Getting better.",
                "The fever and chest tightness are subsiding. Cough still there but it's dry now.",
                "I had pneumonia-type symptoms — productive cough, chest pain, fever — but I'm improving.",
                "I've been recovering from a chest infection. Breathing is normal, cough nearly gone.",
            ],
            "neutral": [
                "Productive cough with yellow-green sputum for {days} day(s). Fever and pleuritic chest pain both resolving.",
                "Bacterial chest infection symptoms — purulent sputum, right-sided chest pain, fever — now improving.",
                "I had productive cough, fever 38.5, and exertional dyspnoea. Duration {days} day(s). Currently resolving.",
                "Chest infection: cough with mucopurulent sputum, fever, pleuritic chest discomfort. Improved over {days} day(s).",
                "Lower respiratory symptoms including purulent sputum, low-grade fever, mild dyspnoea. Now resolving.",
                "I measured SpO2 at 97% today after it was 95% earlier in the week. Cough and fever resolving.",
                "Productive cough with green mucus and right lower chest pain. Temperature elevated. Day {days}, resolving.",
                "Chest infection lasting {days} day(s): fever, productive cough, mild breathing difficulty. Improving.",
                "I had bacterial lower respiratory tract symptoms — purulent sputum, fever, chest pain. Resolving now.",
                "Fever and productive cough with pleuritic component. {days} day(s) duration, now improving.",
                "I had significant chest symptoms with purulent sputum and fever over {days} day(s). Now recovering.",
                "Febrile illness with productive cough — yellow sputum, right-sided chest pain. Duration {days} day(s), resolving.",
                "Lower respiratory tract infection features now resolving: purulent cough, fever, dyspnoea.",
                "I had chest tightness, productive cough, and fever for {days} day(s). Symptoms subsiding.",
                "Bacterial chest infection picture, improving. Purulent cough and chest pain diminishing.",
            ],
            "anxious": [
                "I had a really worrying cough with yellow phlegm and I want to make sure my chest infection has fully cleared.",
                "I seem to be getting better but the productive cough was really frightening — I want confirmation it's resolved.",
                "My cough is almost gone but I had green phlegm and chest pain and I want a doctor to confirm I'm okay.",
                "I've had a chest infection for {days} day(s) and although I'm improving I'm worried about complications.",
                "I was really scared when I started coughing up colored mucus. It's getting better but I want to be checked.",
                "The chest pain when I was breathing was awful and I want to make absolutely sure it's resolved.",
                "I've been measuring my oxygen with a home oximeter — it was 95% and now it's 97%. Am I clear?",
                "I'm improving from a chest infection but chest infections make me very nervous. I need reassurance.",
                "My doctor put me on antibiotics but I want to come back to make sure the infection has actually cleared.",
                "I had productive cough and fever and some breathing difficulty. I think I'm better but I'm not 100% sure.",
                "I've been very anxious about my breathing since having a chest infection. I want a proper check.",
                "The coughing up of yellow mucus and chest pain was very alarming. I need to know I'm fully recovered.",
                "I'm probably fine but after {days} day(s) of a chest infection with breathing difficulty I need reassurance.",
                "My breathing feels almost normal but I keep listening to my chest and I'm worried I can still hear something.",
                "I want to confirm that the pneumonia is fully resolved — my symptoms are mostly gone but I'm still nervous.",
            ],
        },

        "mild": {
            "stoic": [
                "I've had a productive cough with some yellow mucus and a mild fever for {days} day(s). Getting by.",
                "There's some chest tightness and a cough with phlegm. Mild, manageable. {days} day(s).",
                "I've been coughing up phlegm — yellowish — for {days} day(s) with a mild temperature. Nothing dramatic.",
                "Mild chest infection I think. Productive cough, low fever, a bit of chest pain when I breathe deeply.",
                "I've had a chesty cough with some mucus and a temperature for about {days} day(s). Annoying but manageable.",
                "There's some discomfort in my chest especially when I take a deep breath. Productive cough too.",
                "I've been getting a bit short of breath on exertion and I have a productive cough. Low-grade fever.",
                "I don't want to make a fuss but the cough with yellow phlegm has been there for {days} day(s).",
                "Mild chest pain, productive cough, low temperature. I've had worse. {days} day(s) now.",
                "Cough bringing up some yellowish mucus, mild fever, a bit of chest tightness. That's about it.",
                "I've been coughing more than usual with some phlegm and there's a dull ache in my chest.",
                "Low-grade fever and productive cough for {days} day(s). No severe breathing difficulty.",
                "I have a mild chest infection — cough, phlegm, slight temperature. Functioning but not well.",
                "Some right-sided chest ache that's worse on inspiration and a cough with mucus. Mild.",
                "Productive cough with mild fever for {days} day(s). A bit breathless going up stairs.",
            ],
            "neutral": [
                "Mild productive cough with mucopurulent sputum. Low-grade fever 37.8°C. Mild pleuritic chest pain. {days} day(s).",
                "Bacterial chest infection: productive cough, fever 38°C, mild dyspnoea on exertion. Duration {days} day(s).",
                "Cough producing yellow sputum with right-sided chest pain. Temperature 37.9. Mild case. {days} day(s).",
                "I have lower respiratory symptoms: productive cough, low fever, mild pleuritic chest discomfort. Day {days}.",
                "Mild pneumonic illness: fever 38°C, productive cough, SpO2 96% on room air. Duration {days} day(s).",
                "Productive cough with mucopurulent sputum and pleuritic chest pain. Temperature 38.0. {days} day(s).",
                "I have a chest infection: cough with yellow/green sputum, fever, mild shortness of breath on exertion.",
                "Lower respiratory tract infection features. Fever 37.8, productive cough, chest pain on deep inspiration.",
                "Fever 38.2, chest pain that is worse on breathing, productive cough. Started {days} day(s) ago.",
                "Mild pneumonia symptoms: productive cough, low-grade fever, exertional breathlessness. Day {days}.",
                "Chest infection: mucopurulent cough, temperature 38°C, mildly reduced exercise tolerance. {days} day(s).",
                "I have a productive cough with yellow sputum and right lower chest discomfort. Temperature 38.0.",
                "Mild chest infection: fever, cough with sputum, pleuritic pain. Duration {days} day(s). SpO2 97%.",
                "Temperature 37.9, productive cough, right-sided chest ache worse on inspiration. {days} day(s) duration.",
                "Mild bacterial chest infection: low-grade fever, productive cough, mild breathlessness.",
            ],
            "anxious": [
                "I've been coughing up yellow mucus for {days} day(s) and I'm quite worried about a chest infection.",
                "I have a productive cough with green phlegm and it's really alarming me — could this be pneumonia?",
                "There's chest pain when I breathe in and I'm producing phlegm. I'm quite scared — I want this checked.",
                "I've had a chesty cough with colored mucus for {days} day(s) and I'm worried it's getting into my lungs.",
                "The chest tightness and productive cough have me really anxious. What if it's pneumonia?",
                "I've been a bit breathless and coughing up yellow phlegm — I want a chest X-ray to be safe.",
                "My cough is productive and my chest hurts when I breathe deeply. I'm worried about a serious infection.",
                "I measured my oxygen at home — it was 96%. Is that normal? I also have a cough with mucus and a fever.",
                "I'm scared of chest infections going to pneumonia. I have a productive cough and low fever — please check.",
                "The chest pain on inspiration and productive cough are worrying me. I'd like to be thoroughly checked.",
                "I've been coughing up phlegm and feeling feverish for {days} day(s). I'm anxious this could be serious.",
                "I've had bad chest infections before that needed hospital — I want to get this one seen immediately.",
                "The yellow mucus in my cough is alarming me. I also have chest discomfort. I need reassurance.",
                "I'm quite frightened about my breathing — slightly short of breath and my cough is producing mucus.",
                "I want to be checked carefully — I have a productive cough and chest pain and I'm anxious it's pneumonia.",
            ],
        },

        "moderate": {
            "stoic": [
                "I've been fairly unwell with a chest infection for {days} day(s). Productive cough, chest pain, temperature.",
                "I've had proper chest pain — worse when I breathe — and I've been coughing up significant phlegm.",
                "My breathing is definitely affected. Productive cough with green sputum, chest pain, fever over 38.",
                "I'm not at full capacity. Chest infection with significant cough, chest tightness, and a reasonable fever.",
                "I've been coughing a lot — bringing up green-yellow mucus — and there's notable chest pain. {days} day(s).",
                "My breathing is laboured and I have a productive cough. Fever's been around 38.5.",
                "I've had to take time off work. Chest infection — productive cough, chest pain, breathing affected.",
                "I'm noticeably short of breath and I'm coughing up mucus. The chest pain is quite uncomfortable.",
                "This chest infection is affecting my breathing more than I'd like. Temperature 38.8, productive cough.",
                "I've been moderately unwell — coughing up significant phlegm, chest hurts to breathe. {days} day(s).",
                "The chest infection has been knocking me about. Productive cough, pleuritic pain, fever.",
                "I'm managing but this chest infection is significant. Breathing is reduced, cough is very productive.",
                "I've had {days} day(s) of a chest infection with considerable chest pain and difficulty breathing deeply.",
                "I've been producing a lot of yellow-green sputum and the chest pain is worse when I cough.",
                "This chest infection is worse than I anticipated. Breathing difficulty, productive cough, fever 39°C.",
            ],
            "neutral": [
                "Fever 38.8°C, productive cough with copious purulent sputum, pleuritic chest pain. Duration {days} day(s).",
                "Moderate pneumonia presentation: fever 39°C, productive cough, dyspnoea at rest, SpO2 94%. Day {days}.",
                "I have a chest infection with fever 38.7, right-sided pleuritic chest pain, purulent cough, mild dyspnoea.",
                "Moderate bacterial pneumonia: purulent cough, fever 38.9, inspiratory chest pain, reduced exercise tolerance.",
                "Fever 38.8, productive cough with green sputum, pleuritic right lower lobe pain. SpO2 95%. Day {days}.",
                "I have breathlessness at rest, productive cough with purulent sputum, and a fever of 39°C.",
                "Significant lower respiratory infection: fever, pleuritic chest pain, productive cough. SpO2 94-95%.",
                "Temperature 39°C. Purulent cough, pleuritic chest discomfort, dyspnoea on minimal exertion. {days} day(s).",
                "I have moderate chest infection features: fever, purulent sputum, chest pain worse on inspiration.",
                "Productive cough with green phlegm, pleuritic chest pain, fever 38.9. Duration {days} day(s).",
                "I measured SpO2 at 94% today. Productive cough, fever, chest pain. Moderately unwell.",
                "Bacterial pneumonia picture: fever 38.8, right lower chest pain, purulent cough, dyspnoea. Day {days}.",
                "Moderate respiratory illness: purulent cough, fever 39°C, pleuritic pain, reduced SpO2.",
                "I have a fever of 38.7, significant chest pain on breathing, and I'm producing green sputum daily.",
                "Lower respiratory tract infection at moderate severity. Fever, purulent cough, dyspnoea, pleuritic pain.",
            ],
            "anxious": [
                "I'm very worried — my breathing is affected and I'm coughing up green phlegm. I measured SpO2 at 94%.",
                "The chest pain when I breathe is frightening me and I'm producing a lot of yellow-green mucus.",
                "I'm scared — my oxygen was 94% on my home monitor. I have a cough with green phlegm and chest pain.",
                "I feel like I'm not breathing properly and I'm coughing up significant phlegm. This is worrying me a lot.",
                "My chest hurts every time I breathe and I'm producing green mucus. Please take this seriously.",
                "I've been short of breath for {days} day(s) and the cough with purulent sputum is really alarming me.",
                "I have a chest infection and I'm scared it's pneumonia — SpO2 94%, green phlegm, chest pain.",
                "The pleuritic chest pain is quite severe and I'm producing copious yellow mucus. I'm very worried.",
                "I'm frightened about my breathing — it's laboured at rest and I have a very productive cough.",
                "I think I need a chest X-ray — I have a productive cough, chest pain, and I'm having trouble breathing.",
                "I've been tracking my oxygen — it dropped to 93% last night. I'm scared. Also productive cough and fever.",
                "My breathing feels restricted and the chest pain on inspiration is alarming. I want urgent assessment.",
                "I'm very anxious — I can hear something in my chest and I'm coughing up significant green phlegm.",
                "I need to be taken seriously — I have a fever, pleuritic chest pain, and green sputum. Please help.",
                "I'm scared this chest infection is becoming serious — poor breathing, purulent cough, chest pain.",
            ],
        },

        "severe": {
            "stoic": [
                "I can't breathe properly. I've been coughing up a lot of phlegm and the chest pain is severe.",
                "My breathing is significantly compromised. I've needed help getting here. Severe chest pain, productive cough.",
                "I won't understate it — I'm struggling to breathe properly. Very productive cough, high fever, chest pain.",
                "I've been unable to care for myself for {days} day(s). Severe shortness of breath, coughing up bloody phlegm.",
                "My oxygen was 91% at home. I have severe chest pain on breathing and a very productive cough.",
                "This chest infection has become serious. I'm very short of breath even at rest. Had to be brought in.",
                "I can barely get enough air. Very productive cough, severe pleuritic pain, high fever.",
                "I'm genuinely struggling with breathing. Severe chest infection. Someone brought me in.",
                "I've had {days} day(s) of severe chest infection. I can barely breathe and the pain is severe.",
                "My breathing is bad — I needed to come immediately. Very productive cough, chest pain, low oxygen.",
                "I'm in serious trouble with my chest. Severe shortness of breath, I'm coughing up bloody phlegm.",
                "I have a very severe chest infection. I can't breathe well, the pain is constant, high fever.",
                "I struggled to get here — so short of breath. Very productive cough, fever 39.5, chest pain throughout.",
                "I've been fighting a bad chest infection for {days} day(s). Breathing is now severely affected.",
                "My SpO2 hit 90% at home. Severe breathing difficulty, very productive purulent cough, severe chest pain.",
            ],
            "neutral": [
                "Severe pneumonia: SpO2 90-91% on room air, fever 39.5°C, respiratory rate elevated, purulent cough. Day {days}.",
                "I have a severe chest infection with SpO2 91%, high fever, significant dyspnoea, purulent cough with blood.",
                "Severe lower respiratory infection: SpO2 91%, fever 39.8°C, unable to speak full sentences, productive cough.",
                "SpO2 90% at home. Severe pleuritic chest pain, copious purulent sputum, fever 39.5°C. Day {days}.",
                "Severe pneumonia presentation: fever 39.6, respiratory rate 24, SpO2 91%, purulent cough, severe pain.",
                "I have severe dyspnoea with SpO2 of 91%, very productive purulent cough, and fever 39.5°C.",
                "Severe bacterial pneumonia: SpO2 90%, high fever, severe pleuritic pain, bloodstained sputum. Day {days}.",
                "Breathing severely compromised — SpO2 90%, elevated respiratory rate, high fever, purulent cough.",
                "I measured SpO2 of 90-91% at home. Very severe chest infection — purulent cough, fever, severe chest pain.",
                "Severe lower respiratory tract infection: SpO2 91%, fever 39.8°C, dyspnoea at rest, purulent cough.",
                "I need urgent assessment — SpO2 90%, very productive cough with purulent sputum, severe chest pain.",
                "Severe pneumonic illness: high fever 39.5, SpO2 91%, severe dyspnoea, copious purulent sputum.",
                "Temperature 39.7°C, SpO2 90-91%, severe pleuritic chest pain, very productive cough. Day {days}.",
                "I have a SpO2 of 90% and severe chest pain with every breath. Very productive purulent cough.",
                "Severe respiratory compromise: SpO2 91%, fever, purulent cough with haemoptysis, severe chest pain.",
            ],
            "anxious": [
                "I'm terrified — my oxygen is at 90% on my home monitor. I can barely breathe. Please help me.",
                "I can't breathe properly and my oxygen was 90% — I came straight in. I have a very productive cough.",
                "I'm very scared — severe chest pain when I breathe, SpO2 90%, coughing up bloody phlegm.",
                "Please help me — I can barely take a breath without severe pain. My oxygen is very low.",
                "I'm really frightened about my breathing. I measured 91% oxygen and the chest pain is unbearable.",
                "I've been terrified for {days} day(s) — very short of breath, coughing up phlegm with blood.",
                "I'm scared I'm going to stop breathing — severe pain, very low oxygen, very productive cough.",
                "My oxygen dropped to 90% and I'm in so much pain with every breath. I came in immediately.",
                "I'm trembling with fear — I can barely breathe, my chest pain is severe, and I'm coughing up phlegm.",
                "Please take this seriously — SpO2 90%, severe chest pain on breathing, very productive purulent cough.",
                "I've been monitoring my oxygen closely — it was 90% and I'm in terrible pain with every breath.",
                "I'm very frightened about this — I've never had breathing this bad. SpO2 90%, severe chest pain.",
                "I'm genuinely scared for my life — can barely breathe, oxygen very low, coughing up blood-stained phlegm.",
                "I need urgent help — SpO2 is 90% on my monitor and the chest pain is stopping me breathing normally.",
                "I'm extremely anxious — severe shortness of breath, very low oxygen, I could barely get here.",
            ],
        },

        "critical": {
            "stoic": [
                "My breathing is critical. I can't get enough air. SpO2 was 88% when I measured it. I came in urgently.",
                "I need help immediately — I cannot breathe properly. Very low oxygen, severe chest pain.",
                "I've deteriorated rapidly. SpO2 below 90%, barely able to speak, needed ambulance.",
                "This is a medical emergency. My breathing has failed — SpO2 87%, I can't catch my breath.",
                "I've been brought in by ambulance. SpO2 88%, I cannot speak without pausing, severe chest pain.",
                "I can't breathe. SpO2 87%, chest pain every breath, high fever, I needed immediate help.",
                "I'll be direct — I can barely breathe. SpO2 below 88%, very unwell, came by ambulance.",
                "My oxygen has dropped to dangerous levels. I need immediate respiratory assessment.",
                "I can barely form sentences — shortness of breath is that severe. SpO2 87%.",
                "Emergency. SpO2 88%, severe dyspnoea, coughing up blood, fever 40°C. Ambulance brought me in.",
            ],
            "neutral": [
                "Critical pneumonia: SpO2 87-88% on room air, respiratory rate >30, fever 40°C, unable to complete sentences.",
                "Emergency pneumonia presentation: SpO2 88%, severe dyspnoea, haemoptysis, fever 40°C. Ambulance transport.",
                "SpO2 87%, respiratory failure developing. Severe pneumonia with haemoptysis and fever 40°C. Day {days}.",
                "Critical respiratory compromise: SpO2 87-88%, RR 32, fever 40°C, copious purulent and bloodstained sputum.",
                "I have critical pneumonia — SpO2 87%, unable to speak in full sentences, high fever, severe chest pain.",
                "Acute critical pneumonia: SpO2 88%, severe dyspnoea, haemoptysis, temperature 40°C. Ambulance.",
                "Emergency presentation: SpO2 87%, fever 40.2°C, respiratory distress, bloody purulent sputum.",
                "Critical lower respiratory failure: SpO2 88%, RR elevated, fever 40°C, severe dyspnoea.",
                "SpO2 87-88% at home, brought in by ambulance. Fever 40°C, unable to breathe without distress.",
                "Severe pneumonia with respiratory failure: SpO2 87%, fever 40°C, haemoptysis, severe dyspnoea.",
            ],
            "anxious": [
                "I'm terrified — I can barely breathe, my oxygen is 88% and I don't know if I'll make it. Please help.",
                "I came by ambulance because I genuinely thought I was dying — SpO2 88%, can barely breathe.",
                "I'm absolutely terrified — my SpO2 was 87% at home and I cannot breathe. I think I need intensive care.",
                "Please help me right now — I can barely speak I'm so short of breath. My oxygen is critically low.",
                "I'm frightened I'm going to die — SpO2 87%, severe chest pain, I'm coughing up blood.",
                "I came in by ambulance, I was too frightened to wait. SpO2 88%, barely breathing, high fever.",
                "I've never been this scared. I can barely breathe, my oxygen dropped to 87%, I'm in terrible pain.",
                "Please help me — I can't breathe, my SpO2 is 88%, I'm coughing up blood. I'm terrified.",
                "I need help now — I cannot breathe properly, oxygen very low, severe chest pain. I'm in crisis.",
                "I'm desperate — SpO2 87%, I can't speak without stopping to breathe, I'm coughing blood. Please help.",
            ],
        },
    },  # end pneumonia

    # ══════════════════════════════════════════════════════════════════════════
    # NON-INFECTIOUS (worried well, stress, anxiety, minor ailments)
    # Hallmarks: absent or very low fever, non-specific fatigue, no respiratory
    # dominance, often stress/lifestyle component, mild or self-resolving.
    # ══════════════════════════════════════════════════════════════════════════
    "non_infectious": {

        "discharge": {
            "stoic": [
                "I was a bit run down last week but I'm completely fine now. Probably just needed rest.",
                "I thought I might be coming down with something but it came to nothing. Just a bit tired.",
                "I'm feeling okay now. I was a bit off for a couple of days — fatigue, no energy — but it's passed.",
                "I'm not sure why I came in really. I was tired and had a mild headache but it's resolved.",
                "I had a couple of days where I felt run down and a bit under the weather. All fine now.",
                "I don't think there's anything wrong. I felt a bit off but no fever, no cough, nothing specific.",
                "I've recovered from what was essentially just fatigue. No infectious symptoms at any point.",
                "I had some mild symptoms — headache, tiredness — that turned out to be nothing serious.",
                "I'm well. I had a brief period of feeling off but no fever, no real symptoms. Probably just stress.",
                "I came in because I felt odd for a couple of days. No specific symptoms now, all normal.",
                "I was run down for a few days. No temperature, no cough. I think I just needed to sleep more.",
                "All fine now. Had some fatigue and a mild headache but it passed without treatment.",
                "I felt vaguely unwell for a couple of days — nothing specific, no fever. All clear now.",
                "I'm here just to be thorough. Had mild fatigue and some mild nausea but no real illness.",
                "I was tired and had some headaches but no specific infectious symptoms at all.",
            ],
            "neutral": [
                "Brief self-limiting episode of fatigue and mild headache. No fever at any point. Resolved.",
                "I had a non-specific malaise lasting {days} day(s). No fever, no respiratory symptoms. Now well.",
                "Mild fatigue and tension headache for {days} day(s). Temperature normal throughout. Resolved.",
                "I experienced non-specific fatigue and poor sleep without fever or localising symptoms. Now resolved.",
                "Self-limiting episode of fatigue and mild nausea. No fever, no cough, no infection markers.",
                "I had a brief period of low energy and headache. Temperature was 36.8 throughout. Now normal.",
                "Non-specific malaise, no fever, no respiratory symptoms, no localising features. Resolved.",
                "I had mild fatigue and generalised aches without fever or productive cough. Self-limiting.",
                "I experienced {days} day(s) of mild fatigue. No fever, no chest symptoms, no specific diagnosis.",
                "Brief episode of non-specific tiredness and headache. Normal temperature. Resolved.",
                "Self-limiting fatigue and minor headache. No elevated temperature, no infectious features.",
                "I had non-specific malaise for {days} day(s). No fever, no respiratory involvement. Now resolved.",
                "Low energy and mild GI discomfort for {days} day(s). No fever. Self-limited.",
                "Minor self-limiting illness: fatigue, mild headache. Temperature normal. Resolved.",
                "Non-specific episode, no fever, no respiratory symptoms, resolved without treatment.",
            ],
            "anxious": [
                "I felt a bit off and I wanted to be absolutely sure nothing was wrong. I have health anxiety.",
                "I was tired and had a mild headache and I kept thinking it might be something serious.",
                "I know I'm probably fine but I felt off for a couple of days and I needed reassurance.",
                "I had some fatigue and I was convinced it was something infectious — I needed someone to check.",
                "I've been anxious about every symptom since I read about the local outbreak. I felt run down.",
                "I know it's probably nothing but I couldn't rest until I'd been checked. Just fatigue and headache.",
                "My anxiety makes me come in when I feel off. I had mild fatigue and no fever but I was worried.",
                "I felt unwell and I catastrophise — I thought the worst. It's probably just stress and poor sleep.",
                "I keep thinking every little symptom is serious. I had some tiredness and a headache.",
                "I wasn't sleeping well and I felt run down and I started worrying it was something catching.",
                "I know logically I'm probably fine but mild fatigue and headaches make me very anxious.",
                "I'm a worrier. I had some fatigue and I immediately thought I was getting ill. Wanted to check.",
                "I couldn't stop worrying about whether I was getting sick. Fatigue, headache, no fever.",
                "I feel embarrassed coming in for this but I was really worried. Just fatigue and mild nausea.",
                "My health anxiety gets the better of me — I had mild symptoms and needed professional reassurance.",
            ],
        },

        "mild": {
            "stoic": [
                "I've had a mild headache and some fatigue for {days} day(s). No fever, no cough.",
                "I'm a bit run down. Tired, some muscle tension, not sleeping well. No infectious symptoms.",
                "I've been under stress and I think it's catching up with me. Just fatigue and a dull headache.",
                "Mild nausea and fatigue for {days} day(s). No vomiting, no fever. Just feeling a bit off.",
                "I've had a nagging tension headache and low energy for {days} day(s). No temperature.",
                "I've been tired and a bit achy but no fever and no real respiratory symptoms.",
                "Low energy and mild stomach discomfort for {days} day(s). I don't think it's anything infectious.",
                "I'm feeling a bit flat. Fatigue and headache. No fever. Probably stress-related.",
                "Some mild GI issues and fatigue for {days} day(s). No fever, no cough.",
                "I've been tired, slightly nauseous, with mild headache. Temperature normal the whole time.",
                "I feel mildly unwell but there's no obvious infection. Fatigue and a tension headache.",
                "I've been run down for {days} day(s). No fever, no respiratory symptoms. Probably just overworked.",
                "Mild fatigue and headache. No temperature, no cough, no specific symptom.",
                "I'm a bit under the weather. Mild stomach upset and fatigue. I don't think I'm infectious.",
                "I've had low energy and a mild headache for {days} day(s). Nothing specific.",
            ],
            "neutral": [
                "Non-infectious presentation: mild fatigue, tension headache, normal temperature. Duration {days} day(s).",
                "Mild fatigue and headache without fever or respiratory symptoms. {days} day(s) duration.",
                "I have mild non-specific malaise: fatigue, headache, low-grade nausea. Temperature 36.9. Day {days}.",
                "Non-specific mild illness: fatigue, mild musculoskeletal aches, no fever, no respiratory signs.",
                "I have mild fatigue and tension-type headache. Temperature normal. No infectious features.",
                "Mild generalised malaise: low energy, headache, mild nausea. No fever. Duration {days} day(s).",
                "Non-infectious symptom picture: fatigue, mild headache, normal temperature, no cough.",
                "I have mild fatigue and headache. Temperature 36.8 throughout. No specific infectious aetiology.",
                "Mild non-specific symptoms: fatigue, tension headache, mild nausea. No fever. Day {days}.",
                "I have fatigue and mild headache without fever or respiratory symptoms. Duration {days} day(s).",
                "Non-specific malaise: low energy, mild headache, no fever, no cough. {days} day(s).",
                "Mild fatigue, tension headache, mild nausea. Normal temperature. Likely stress-related.",
                "I have a mild illness: fatigue, headache, no fever. Duration {days} day(s). No respiratory involvement.",
                "Non-specific mild symptoms without fever or infection signs. Fatigue and headache. Day {days}.",
                "Mild malaise: fatigue, headache, normal temperature. Non-infectious pattern. {days} day(s).",
            ],
            "anxious": [
                "I've had a headache and fatigue for {days} day(s) and I'm worried it could be something contagious.",
                "I keep checking my temperature but it's normal — yet I feel off with headache and fatigue.",
                "I'm quite anxious — I have fatigue and a mild headache and I can't stop thinking about what it could be.",
                "I've been feeling mildly unwell for {days} day(s) with no fever and I'm worried that's suspicious.",
                "I have a persistent mild headache and fatigue. I know it's probably nothing but I need to check.",
                "I've been reading about infectious diseases and now every little symptom worries me. Just fatigue and headache.",
                "I can't stop thinking there might be something wrong — I have fatigue and mild nausea but no fever.",
                "I've taken my temperature six times today — all normal — but I still feel off. Headache, fatigue.",
                "I'm here because I'm worried, not because my symptoms are severe. Fatigue and headache, no fever.",
                "I'm catastrophising again — I have mild fatigue and a headache and I'm worried it's something serious.",
                "I need reassurance that this is just stress — fatigue and headache for {days} day(s), no fever.",
                "I've been Googling my symptoms and I've scared myself. It's probably just fatigue and a headache.",
                "I keep worrying I'm getting ill — fatigue, mild headache, no fever. I wanted to be checked.",
                "I'm anxious about my health generally and mild symptoms make me very worried. Fatigue and headache.",
                "I couldn't relax at home with these mild symptoms. I know it's probably nothing but I needed to be seen.",
            ],
        },
    },  # end non_infectious
}


_SEVERITY_TO_SIGMA = {
    "discharge": 0.10,
    "mild":      0.35,
    "moderate":  0.60,
    "severe":    0.82,
    "critical":  0.95,
}

_TREND_BY_SEVERITY = {
    "discharge": "improving",
    "mild":      "stable",
    "moderate":  "stable",
    "severe":    "worsening",
    "critical":  "worsening",
}


class _MockInnerState:
    """Minimal stand-in for InnerState so SymptomNarrator can run without simulation."""
    def __init__(self, severity: float, disease_name: str, trend: str):
        self.severity     = severity
        self.disease_name = disease_name
        self.trend        = trend
        self.top_vital    = None   # skip vital addon for simplicity


class TemplateSampler:
    """
    Uses `SymptomNarrator.full_opening_statement` to generate text.
    Same interface as PhraseLibrary; no hand-written phrases.

    Disease-distinguishing phrases come from INFLUENZA_PHRASES / PNEUMONIA_PHRASES
    in symptom_language.py.  Lower lexical diversity than PhraseLibrary but faster
    to set up and a clean baseline for the phrase-bank comparison.
    """

    def __init__(self, seed: int = 42, confusion_rate: float = 0.0):
        from simulation.symptom_language import SymptomNarrator, Personality
        import random
        self._rng            = random.Random(seed)
        self._narrator       = SymptomNarrator(rng=self._rng)
        self._Personality    = Personality
        self._confusion_rate = confusion_rate

    def sample(
        self,
        disease: str,
        severity: str,
        personality: str = "neutral",
        days: Optional[int] = None,
    ) -> dict:
        if days is None:
            days = self._rng.randint(1, 14)

        # With probability confusion_rate, generate text as if the patient
        # had a different disease (atypical presentation).  Ground-truth label
        # stays correct so the irreducible error floor ≈ confusion_rate.
        text_disease = disease
        if self._confusion_rate > 0 and self._rng.random() < self._confusion_rate:
            others = [d for d in _SEVERITY_TO_SIGMA if d != disease]  # wrong disease
            # pick from same-severity neighbours
            all_dis = ["influenza", "pneumonia", "non-infectious"]
            others  = [d for d in all_dis if d != disease]
            text_disease = self._rng.choice(others)

        sigma      = _SEVERITY_TO_SIGMA.get(severity, 0.5)
        trend      = _TREND_BY_SEVERITY.get(severity, "stable")
        p_enum     = getattr(self._Personality, personality.upper(),
                             self._Personality.NEUTRAL)
        inner      = _MockInnerState(sigma, text_disease, trend)
        text       = self._narrator.full_opening_statement(inner, days, p_enum)
        label      = f"{disease}/{severity}"
        return {
            "text":        text,
            "label":       label,
            "gt_disease":  disease,
            "gt_severity": severity,
            "days":        days,
        }

    def sample_batch(
        self,
        disease: str,
        severity: str,
        n: int,
        personality: str = "neutral",
    ) -> list[dict]:
        return [self.sample(disease, severity, personality) for _ in range(n)]

    def available_buckets(self) -> list[tuple[str, str]]:
        return list(_SEVERITY_TO_SIGMA.keys())


class PhraseLibrary:
    """
    Sample a symptom-description text for a given (disease, severity, personality).

    Returns a record dict compatible with FLLearner / SiloDataset:
        {text, label, gt_disease, gt_severity, days}
    """

    # Map numeric severity bands (0/1/2) to label names for compatibility
    _BAND_TO_SEVERITY = {0: "mild", 1: "moderate", 2: "severe"}

    def __init__(self, seed: int = 42, confusion_rate: float = 0.0):
        self._rng            = random.Random(seed)
        self._confusion_rate = confusion_rate

    def sample(
        self,
        disease: str,
        severity: str,
        personality: str = "neutral",
        days: Optional[int] = None,
    ) -> dict:
        """
        Parameters
        ----------
        disease     : "influenza" | "pneumonia" | "non_infectious"
        severity    : "discharge" | "mild" | "moderate" | "severe" | "critical"
        personality : "stoic" | "neutral" | "anxious"
        days        : days-infected integer; random 1-14 if None
        """
        if days is None:
            days = self._rng.randint(1, 14)

        # With probability confusion_rate, generate text from a different disease
        # (atypical presentation).  Ground-truth label stays correct.
        text_disease = disease
        if self._confusion_rate > 0 and self._rng.random() < self._confusion_rate:
            all_dis     = ["influenza", "pneumonia", "non-infectious"]
            text_disease = self._rng.choice([d for d in all_dis if d != disease])

        # Normalize to underscore for PHRASES dict lookup (handles "non-infectious")
        d_key = text_disease.lower().replace(" ", "_").replace("-", "_")
        s_key = severity.lower()
        p_key = personality.lower()

        # Graceful fallback: unknown personality → neutral
        bucket = (
            PHRASES.get(d_key, {})
            .get(s_key, {})
            .get(p_key)
        )
        if bucket is None:
            bucket = (
                PHRASES.get(d_key, {})
                .get(s_key, {})
                .get("neutral", [])
            )
        if not bucket:
            # Last-resort fallback to any non-empty bucket in same disease
            for sev in ("moderate", "mild", "severe", "discharge", "critical"):
                bucket = PHRASES.get(d_key, {}).get(sev, {}).get("neutral", [])
                if bucket:
                    s_key = sev
                    break

        text = self._rng.choice(bucket).format(days=days)
        # Preserve the caller's disease name in output so it matches label maps
        # (e.g. "non-infectious" matches DISEASE_LABELS, not "non_infectious")
        label = f"{disease}/{s_key}"
        return {
            "text":        text,
            "label":       label,
            "gt_disease":  disease,
            "gt_severity": s_key,
            "days":        days,
        }

    def sample_batch(
        self,
        disease: str,
        severity: str,
        n: int,
        personality: str = "neutral",
    ) -> list[dict]:
        return [self.sample(disease, severity, personality) for _ in range(n)]

    def available_buckets(self) -> list[tuple[str, str]]:
        """Return all (disease, severity) pairs present in the library."""
        return [
            (d, s)
            for d, sevs in PHRASES.items()
            for s in sevs
        ]
