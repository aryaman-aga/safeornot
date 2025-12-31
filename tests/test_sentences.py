import requests
import json

# List of sentences to test
sentences = [
    "Welcome! Please introduce yourself and share what you hope to learn here.[1]",
    "Thanks for sharing your experience—this is really helpful for others.",
    "Let’s keep the conversation respectful and focus on the topic.",
    "I can volunteer to moderate the weekly Q&A thread if needed.",
    "Great point—could you add a source so everyone can verify the info?",
    "If you’re new, start with the FAQ and community guidelines.",
    "I disagree, but I appreciate your perspective and want to understand it better.",
    "Please avoid personal attacks; critique ideas, not people.",
    "Happy to help—DM me and I’ll walk you through the setup.",
    "This community is for learning together, so questions are always welcome.",
    "Let’s move this to the right channel so it stays organized.",
    "Congrats on the milestone! Thanks for sharing the lessons learned.",
    "I reported the bug with steps to reproduce so the team can fix it faster.",
    "Could we create a resource list for beginners and keep it pinned?",
    "I appreciate the reminder—I'll edit my post to match the rules.",
    "Let’s welcome newcomers and make sure they feel included.",
    "I can summarize the thread and post key takeaways for others.",
    "Thanks for the feedback—I'll clarify my question to reduce confusion.",
    "Please use inclusive language so everyone feels safe participating.",
    "Before posting, try searching—there are some great answers already.",
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
    "I’m here to pick fights, not to contribute."
]

url = "http://localhost:8000/predict"

print(f"{'SENTENCE':<80} | {'LABEL':<10} | {'CONFIDENCE':<10}")
print("-" * 110)

for sentence in sentences:
    try:
        response = requests.post(url, json={"text": sentence})
        if response.status_code == 200:
            result = response.json()
            label = result['label']
            confidence = result['confidence']
            print(f"{sentence[:77]:<80} | {label:<10} | {confidence:.4f}")
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Connection Error: {e}")
        print("Make sure the server is running: python server.py")
        break
