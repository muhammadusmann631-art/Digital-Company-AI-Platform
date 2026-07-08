"""
Poetry Generator Module
Generates poetry based on detected emotions
"""
from langchain_groq import ChatGroq
import config


class PoetryGenerator:
    """Generate emotion-based poetry using AI"""
    
    def __init__(self):
        """Initialize poetry generator"""
        self.model = ChatGroq(
            model='llama-3.1-8b-instant',
            api_key=config.GROQ_API_KEY,
            temperature=0.9  # Higher temperature for more creative output
        )
    
    def generate_poetry(self, emotion, style='stanza'):
        """
        Generate poetry based on emotion (legacy method for backward compatibility)
        Args:
            emotion: Detected emotion (angry, sad, happy, etc.)
            style: Poetry style ('stanza', 'haiku', 'free_verse')
        Returns:
            Generated poem as string
        """
        result = self.generate_bilingual_poetry(emotion)
        return result['english'] if result else "Unable to generate poetry."
    
    def generate_bilingual_poetry(self, emotion):
        """
        Generate bilingual poetry (English + Urdu) based on emotion
        Args:
            emotion: Detected emotion (angry, sad, happy, etc.)
        Returns:
            dict with 'english' and 'urdu' poems
        """
        # Create emotion-specific prompts
        emotion_themes = {
            'angry': {
                'english': "anger, rage, frustration, fire, storm, thunder",
                'urdu': "غصہ، طیش، جوش، آگ، طوفان"
            },
            'sad': {
                'english': "sadness, sorrow, tears, loneliness, melancholy, rain",
                'urdu': "غم، اداسی، آنسو، تنہائی، ویرانی"
            },
            'happy': {
                'english': "happiness, joy, celebration, sunshine, laughter, bliss",
                'urdu': "خوشی، مسرت، جشن، روشنی، ہنسی"
            },
            'fear': {
                'english': "fear, anxiety, darkness, shadows, uncertainty, trembling",
                'urdu': "خوف، اندیشہ، اندھیرا، سایہ، لرزش"
            },
            'surprise': {
                'english': "surprise, wonder, amazement, discovery, revelation, awe",
                'urdu': "حیرت، تعجب، انکشاف، کرشمہ"
            },
            'disgust': {
                'english': "disgust, aversion, rejection, bitterness, disdain",
                'urdu': "نفرت، کراہت، تلخی، انکار"
            },
            'neutral': {
                'english': "peace, calm, balance, serenity, stillness, reflection",
                'urdu': "سکون، امن، توازن، خاموشی، فکر"
            }
        }
        
        theme = emotion_themes.get(emotion, emotion_themes['neutral'])
        
        # Generate English poem
        english_prompt = f"""Write a beautiful 5-stanza poem about {emotion}. 
        Each stanza must have exactly 4 lines.
        Use themes like: {theme['english']}
        Make it rhyme (ABAB or AABB pattern).
        Be creative, emotional, and poetic.
        Only return the poem, no title or extra text."""
        
        # Generate Urdu poem
        urdu_prompt = f"""اردو میں ایک خوبصورت نظم لکھیں جو {emotion} کے جذبات کو بیان کرے۔
        نظم میں 5 بند ہونے چاہئیں۔
        ہر بند میں 4 مصرعے ہوں۔
        ہر مصرعے کے بعد لائن بریک (line break) استعمال کریں۔
        ہر بند کے بعد دو لائن بریک استعمال کریں۔
        موضوعات: {theme['urdu']}
        ردیف اور قافیہ استعمال کریں۔
        صرف نظم لکھیں، عنوان یا اضافی متن نہیں۔"""
        
        try:
            # Generate English poem
            english_response = self.model.invoke(english_prompt)
            english_poem = english_response.content.strip()
            
            # Generate Urdu poem
            urdu_response = self.model.invoke(urdu_prompt)
            urdu_poem = urdu_response.content.strip()
            
            return {
                'english': english_poem,
                'urdu': urdu_poem
            }
            
        except Exception as e:
            return {
                'english': f"Unable to generate English poetry.\nError: {str(e)}",
                'urdu': f"اردو شاعری بنانے میں خرابی۔\nخرابی: {str(e)}"
            }
    
    def get_emotion_description(self, emotion):
        """Get a description of the emotion"""
        descriptions = {
            'angry': "You seem to be feeling anger or frustration right now.",
            'sad': "You appear to be feeling sad or melancholic.",
            'happy': "You look happy and joyful!",
            'fear': "You seem to be experiencing fear or anxiety.",
            'surprise': "You appear surprised or amazed!",
            'disgust': "You seem to be feeling disgust or aversion.",
            'neutral': "You have a calm, neutral expression."
        }
        return descriptions.get(emotion, "I detected an emotion.")
