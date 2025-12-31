import pandas as pd
import random
import numpy as np

# --- DATA SOURCES ---

safe_english_templates = [
    "Today is a wonderful day to go for a walk in the park.",
    "The weather in Delhi is surprisingly pleasant this evening.",
    "I am going to the market to buy some vegetables and fruits.",
    "Safety is a fundamental concept that we must all adhere to.",
    "Traffic rules are meant to keep us safe on the roads.",
    "I love listening to classical music while working.",
    "The metro ride was quite comfortable today.",
    "My grandmother makes the best aloo parathas.",
    "We should respect our elders and take care of them.",
    "Education is the key to a successful future.",
    "I am planning a trip to Manali next weekend.",
    "Cricket is more than just a sport in India, it's an emotion.",
    "Please make sure to lock the door before you leave.",
    "The meeting has been rescheduled to next Monday.",
    "I need to finish my assignment before the deadline.",
    "Let's go for a movie this Friday night.",
    "The food at that new restaurant was delicious.",
    "I am learning to play the guitar.",
    "Yoga helps in maintaining both physical and mental health.",
    "It is important to stay hydrated during the summer."
]

# --- HARD NEGATIVES (Safe sentences that might confuse the model) ---
safe_hard_negatives = [
    "Today is a good day but you are nice.",
    "Today is a good day but I am tired.",
    "The weather is bad but we are safe.",
    "I saw a fight in the movie but it was fake.",
    "He is killing it on the dance floor!",
    "That joke was bomb!",
    "I will shoot you an email later.",
    "Don't beat yourself up over a small mistake.",
    "This food is dangerously good.",
    "I am dead tired after work.",
    "She is a bombshell of talent.",
    "You are killing me with these jokes.",
    "I attacked the problem with full force.",
    "The traffic was murder today.",
    "Today is a good day but the traffic is bad.",
    "Mausam kharab hai par hum safe hain.",
    "Woh dance floor par aag laga raha hai (metaphorically).",
    "Main tumhe email shoot kar dunga.",
    "Goli ki raftaar se gaadi chala raha tha woh (fast driving).",
    "Aaj ka din bura tha par ab sab theek hai.",
    "Today is a good day but you are nice, lets go for a walk and have ice cream.",
    "Today is a good day but you are nice, lets go for a walk and talk.",
    "Today is a good day but you are nice, lets go for a walk and play cricket.",
    "Today is a good day but you are nice, lets go for a walk and eat.",
    "Mausam achha hai par tum acche ho, chalo ghumne chalte hain.",
    "You are nice but I am busy right now.",
    "You are nice and I like your personality.",
    "Today is a good day but you are nice, lets go for a walk and eat ice cream.",
    "Today is a good day but you are nice, lets go for a walk and discuss the project.",
    "Today is a good day but you are nice, lets go for a walk and listen to music.",
    "Today is a good day but you are nice, lets go for a walk and enjoy the weather.",
    "Today is a good day but you are nice, lets go for a walk and exercise.",
    "Today is a good day but you are nice, lets go for a walk and meet my friends.",
    "Today is a good day but you are nice, lets go for a walk and buy groceries.",
    "Today is a good day but you are nice, lets go for a walk and sit in the park.",
    "Today is a good day but you are nice, lets go for a walk and read books."
]

safe_dating_templates = [
    "Let's go for a walk and talk about life.",
    "Do you want to grab coffee sometime?",
    "I really enjoy spending time with you.",
    "Would you like to go to the movies with me?",
    "Let's go for a dinner date.",
    "Can we meet at the park?",
    "I like you, do you want to be friends?",
    "Let's hang out this weekend.",
    "Today is a good day, let's go for a drive.",
    "You are nice, let's go for a walk.",
    "Chalo ghumne chalte hain.",
    "Coffee peene chalein?",
    "Mujhe tumhare saath waqt bitana achha lagta hai.",
    "Kya tum mere saath movie dekhne chalogi?",
    "Park mein milte hain.",
    "Aaj mausam achha hai, long drive pe chalte hain."
]

safe_activities = [
    "talk", "have ice cream", "play cricket", "eat", "discuss the project", 
    "listen to music", "enjoy the weather", "exercise", "meet my friends", 
    "buy groceries", "sit in the park", "read books", "have coffee", "dance",
    "chat", "have fun", "relax", "meditate", "jog", "run"
]

safe_hinglish_templates = [
    "Aaj ka mausam bahut achha hai, ghumne chalte hain.",
    "Delhi mein aaj traffic bahut kam hai, lucky day!",
    "Mummy ne aaj khane mein rajma chawal banaye hain.",
    "Exam ki tension mat le, sab theek ho jayega.",
    "Chalo aaj shaam ko chai peene chalte hain.",
    "Woh movie bahut funny thi, maza aa gaya.",
    "Bhai, notes bhej dena please, kal submission hai.",
    "Aaj office mein bahut kaam tha, thak gaya hoon.",
    "Weekend par kya plan hai? Kahin bahar chalein?",
    "Gaadi dhyaan se chalana, sadak kharab hai.",
    "Mujhe naya phone lena hai, kaunsa best rahega?",
    "Cricket match dekhne ka alag hi maza hai doston ke saath.",
    "Aaj kal pollution bahut badh gaya hai shehar mein.",
    "Metro mein aaj seat mil gayi, miracle ho gaya.",
    "Ghar jaldi aa jana, aaj guests aane wale hain.",
    "Padhai kar le, warna fail ho jayega.",
    "Gym jana shuru kar diya hai maine.",
    "Biryani khane ka mann kar raha hai aaj.",
    "Internet aaj bahut slow chal raha hai.",
    "Sab theek hai, bas thoda busy hoon aajkal."
]

unsafe_keywords_english = [
    "kill", "murder", "rape", "sex", "harass", "beat up", "shoot", "bomb", "terrorist", "attack", "abuse", "naked", "porn"
]

unsafe_keywords_hinglish = [
    "bsdk", "mc", "bc", "madarchod", "bhenchod", "chutiya", "gand", "lauda", "randi", "saale", "kutta", "kamina", "marunga", "cheer dunga"
]

unsafe_english_templates = [
    "I will kill you if you don't listen to me.",
    "She looks so sexy, I want to have sex with her right now.",
    "I hate this group of people, they should be bombed.",
    "He is a terrorist and should be shot dead.",
    "Stop following me or I will beat you up.",
    "Show me your naked pictures.",
    "I will rape you if I see you alone.",
    "You are worthless and should just die.",
    "I am going to attack the school tomorrow.",
    "Send nudes or I will leak your photos."
]

unsafe_hinglish_templates = [
    "Tu bahar mil, tujhe batata hoon bsdk.",
    "Kya maal hai yaar, iske saath sex karne ka mann hai.",
    "Teri maa ki aankh, saale kutte.",
    "Woh ladki randi hai, uske paas mat jana.",
    "Tujhe jaan se maar dunga agar tu wahan gaya toh.",
    "Chutiya samajh rakha hai kya mujhe?",
    "Teri gand tod dunga main.",
    "Bhenchod, tu jaanta nahi main kaun hoon.",
    "Usko utha lenge aaj shaam ko.",
    "Saale ko goli maar denge."
]

subtle_harassment_templates = [
    "You look very pretty today, why don't you come to my room for a private chat?",
    "I know where you live, it would be a shame if something happened.",
    "Nice dress, it would look better on my floor.",
    "You are smart for a girl, usually they are dumb.",
    "Come sit on my lap, I will give you a promotion.",
    "Why are you acting so hard to get? I know you want it.",
    "Your smile is inviting me to do things to you.",
    "Let's meet alone, away from your friends.",
    "I have been watching you, you look great in that window.",
    "Don't tell anyone about our little secret meeting."
]

subtle_harassment_hinglish = [
    "Aaj badi sundar lag rahi ho, akele mein milo na.",
    "Tumhare ghar ka raasta mujhe pata hai, bachke rehna.",
    "Kapde achhe hain, par bina kapdon ke zyada achhi lagogi.",
    "Ladki hoke itna dimaag kaise chala leti ho?",
    "Mere paas aao, tumhe khush kar dunga.",
    "Itna bhav kyu kha rahi ho, mujhe pata hai tumhe pasand hai.",
    "Tumhari hansi mujhe kuch aur hi ishara kar rahi hai.",
    "Doston ko chhod, hum dono kahin akele chalte hain.",
    "Main tumhe roz dekhta hoon, tum notice nahi karti.",
    "Hamari baat kisi ko mat batana, yeh hamara raaz hai."
]

unsafe_solicitation_templates = [
    "lets go for a walk and have sex",
    "we should go to my place and sleep together",
    "wanna hook up after this?",
    "let's have some fun in bed",
    "i want to touch you",
    "send me your nude photos",
    "can i kiss you on the lips",
    "let's make out",
    "i want to see you without clothes",
    "come over tonight, my parents are not home",
    "let's go to the backseat of my car"
]

unsafe_solicitation_hinglish = [
    "chal sex karte hain",
    "mere ghar chal, maza karenge",
    "bistar pe chalte hain",
    "mujhe tere saath sona hai",
    "apni nangi photo bhej",
    "kya hum kiss kar sakte hain?",
    "kapde utaar ke dikha",
    "room pe chalte hain, koi nahi hai wahan",
    "aaja gaadi mein baith, kuch karte hain",
    "aaj raat mere paas aaja"
]

# --- NEW: Toxic / Bullying / Spam Templates ---
unsafe_toxic_templates = [
    "This is stupid—only an idiot would ask that.",
    "Stop posting here; nobody cares about your opinion.",
    "If you’re not like us, you don’t belong in this community.",
    "I’m going to spam this channel until I get what I want.",
    "The moderators are useless and should be fired.",
    "Don’t help them—let them fail and quit.",
    "Your question is a waste of time; learn to read.",
    "I’m only here to promote my product; DM me for links.",
    "Report it if you want—nothing ever changes here.",
    "Everyone here is biased, so I’m going to stir drama.",
    "I’ll keep reposting the same thing until it gets attention.",
    "You’re clueless—stop pretending you know anything.",
    "This community is trash and the people are trash.",
    "Only insiders get help here; newcomers should stay quiet.",
    "I’m going to derail this thread because it’s boring.",
    "Your culture doesn’t fit here, so leave.",
    "I’m canceling the event last minute just to mess with people.",
    "I won’t follow the rules; moderators can’t make me.",
    "Go argue somewhere else—this isn’t a place for your kind of thinking.",
    "I’m here to pick fights, not to contribute.",
    "You are so dumb, just leave.",
    "Nobody likes you here.",
    "Kill yourself.",
    "You are a waste of space.",
    "Shut up and go away.",
    "I hate people like you.",
    "You are pathetic.",
    "This post is garbage.",
    "Delete your account.",
    "You are ugly and stupid."
]

# --- NEW: Safe Professional / Disagreement Templates ---
safe_professional_templates = [
    "I disagree, but I appreciate your perspective and want to understand it better.",
    "That is a valid point, however, I have a different view.",
    "Let's agree to disagree on this topic.",
    "I respectfully disagree with your statement.",
    "Could you please clarify what you mean by that?",
    "I understand where you are coming from, but the data suggests otherwise.",
    "Thank you for sharing your opinion, even though I don't agree.",
    "Let's keep the conversation constructive.",
    "I think there might be a misunderstanding here.",
    "Can we focus on the facts rather than personal opinions?",
    "I appreciate your feedback and will take it into consideration.",
    "It's okay to have different opinions.",
    "Let's discuss this calmly.",
    "I see your point, but have you considered this alternative?",
    "That's an interesting perspective."
]

# --- GENERATION LOGIC ---

def generate_safe_paragraph():
    # Mix of English and Hinglish safe sentences
    
    # 30% chance to generate a "safe complex" sentence that mirrors the unsafe structure
    # This is CRITICAL to fix the "walk and talk" vs "walk and have sex" confusion
    if random.random() < 0.3:
        # Often use the specific prefix that was causing issues
        if random.random() < 0.5:
             start = "Today is a good day but you are nice"
        else:
             start = random.choice(safe_english_templates + safe_hinglish_templates)
        
        connector = random.choice([", ", ". ", " but ", " and "])
        activity = random.choice(safe_activities)
        return f"{start}{connector}lets go for a walk and {activity}."

    # 25% chance to include a "hard negative" (safe but tricky) sentence
    if random.random() < 0.25:
        return random.choice(safe_hard_negatives)
        
    # 15% chance to include a "safe professional disagreement" (to fix false positives on "I disagree")
    if random.random() < 0.15:
        return random.choice(safe_professional_templates)

    num_sentences = random.randint(2, 5)
    sentences = []
    for _ in range(num_sentences):
        rand_val = random.random()
        if rand_val < 0.3:
            sentences.append(random.choice(safe_english_templates))
        elif rand_val < 0.6:
            sentences.append(random.choice(safe_hinglish_templates))
        elif rand_val < 0.8:
            sentences.append(random.choice(safe_dating_templates))
        else:
            sentences.append(random.choice(safe_hard_negatives))
            
    return " ".join(sentences)

def generate_unsafe_paragraph():
    # Can be explicitly unsafe or subtle
    # Weighted choice to prioritize the tricky 'safe_turn_unsafe' case
    types = ['explicit_english', 'explicit_hinglish', 'subtle_english', 'subtle_hinglish', 'mixed', 'safe_turn_unsafe', 'toxic']
    weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3] # 30% chance for toxic/bullying
    
    type_unsafe = random.choices(types, weights=weights, k=1)[0]
    
    if type_unsafe == 'toxic':
        return random.choice(unsafe_toxic_templates)
    
    elif type_unsafe == 'explicit_english':
        base = random.choice(unsafe_english_templates)
        context = generate_safe_paragraph() # Embed in safe context
        return f"{context} {base} {generate_safe_paragraph()}"
    
    elif type_unsafe == 'explicit_hinglish':
        base = random.choice(unsafe_hinglish_templates)
        context = generate_safe_paragraph()
        return f"{context} {base} {generate_safe_paragraph()}"
        
    elif type_unsafe == 'subtle_english':
        base = random.choice(subtle_harassment_templates)
        # Subtle often comes with "nice" sounding words
        prefix = "Today is a good day but "
        return f"{prefix} {base}"
        
    elif type_unsafe == 'subtle_hinglish':
        base = random.choice(subtle_harassment_hinglish)
        prefix = "Mausam achha hai par "
        return f"{prefix} {base}"
        
    elif type_unsafe == 'mixed':
        # Mix safe start with sudden unsafe ending
        start = random.choice(safe_english_templates)
        end = random.choice(unsafe_keywords_hinglish)
        return f"{start} but you are a {end}."
        
    elif type_unsafe == 'safe_turn_unsafe':
        # Safe start, explicit end (The user's specific failure case)
        # "Today is a good day but you are nice, lets go for a walk and have sex"
        start = generate_safe_paragraph()
        if random.random() > 0.5:
            end = random.choice(unsafe_solicitation_templates)
        else:
            end = random.choice(unsafe_solicitation_hinglish)
        
        connector = random.choice([", ", ". ", " but ", " and "])
        return f"{start}{connector}{end}"

def generate_dataset(num_samples=5000):
    data = []
    
    # 50% Safe, 50% Unsafe
    for _ in range(num_samples // 2):
        data.append({
            "text": generate_safe_paragraph(),
            "label": "safe"
        })
        
    for _ in range(num_samples // 2):
        data.append({
            "text": generate_unsafe_paragraph(),
            "label": "not safe"
        })
        
    df = pd.DataFrame(data)
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    return df

if __name__ == "__main__":
    print("Generating dataset...")
    df = generate_dataset(10000) # Generate 10k samples
    df.to_csv("dataset.csv", index=False)
    print(f"Dataset generated with {len(df)} samples. Saved to dataset.csv")
    print(df.head())
    print(df['label'].value_counts())
