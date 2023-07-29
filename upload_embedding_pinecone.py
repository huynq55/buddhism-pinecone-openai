import pinecone
import openai

# Initialize OpenAI
# openai = OpenAI(api_key="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")

# Specify the index name
index_name = "buddhism-sutta"

# Initialize Pinecone
pinecone.init(api_key="23c095af-b6e1-453a-98e4-b09a2804ba46", environment="us-central1-gcp")

# Create a new vector index only if it doesn't already exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=1536,  # dimensionality of text-embedding-ada-002
        metric='cosine',
    )

index = pinecone.Index(index_name)

# Generate Embeddings and Upload
def generate_and_upload(text):
    embeddings = get_embedding(text)

    index.upsert([("AN1.11-20", embeddings, {"sutta": "AN1.11-20", "sutta_name": "Nīvaraṇappahānavagga", "text": text})])

def get_embedding(text, model="text-embedding-ada-002"):
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

text = """
11

“Mendicants, I do not see a single thing that gives rise to sensual desire, or, when it has arisen, makes it increase and grow like the feature of beauty. When you apply the mind irrationally to the feature of beauty, sensual desire arises, and once arisen it increases and grows.”

12

“Mendicants, I do not see a single thing that gives rise to ill will, or, when it has arisen, makes it increase and grow like the feature of harshness. When you apply the mind irrationally to the feature of harshness, ill will arises, and once arisen it increases and grows.”

13

“Mendicants, I do not see a single thing that gives rise to dullness and drowsiness, or, when they have arisen, makes them increase and grow like discontent, sloth, yawning, sleepiness after eating, and mental sluggishness. When you have a sluggish mind, dullness and drowsiness arise, and once arisen they increase and grow.”

14

“Mendicants, I do not see a single thing that gives rise to restlessness and remorse, or, when they have arisen, makes them increase and grow like an unsettled mind. When you have no peace of mind, restlessness and remorse arise, and once arisen they increase and grow.”

15

“Mendicants, I do not see a single thing that gives rise to doubt, or, when it has arisen, makes it increase and grow like irrational application of mind. When you apply the mind irrationally, doubt arises, and once arisen it increases and grows.”

16

“Mendicants, I do not see a single thing that prevents sensual desire from arising, or, when it has arisen, abandons it like the feature of ugliness. When you apply the mind rationally to the feature of ugliness, sensual desire does not arise, or, if it has already arisen, it’s given up.”

17

“Mendicants, I do not see a single thing that prevents ill will from arising, or, when it has arisen, abandons it like the heart’s release by love. When you apply the mind rationally on the heart’s release by love, ill will does not arise, or, if it has already arisen, it’s given up.”

18

“Mendicants, I do not see a single thing that prevents dullness and drowsiness from arising, or, when they have arisen, gives them up like the elements of initiative, persistence, and vigor. When you’re energetic, dullness and drowsiness do not arise, or, if they’ve already arisen, they’re given up.”

19

“Mendicants, I do not see a single thing that prevents restlessness and remorse from arising, or, when they have arisen, gives them up like peace of mind. When your mind is peaceful, restlessness and remorse do not arise, or, if they’ve already arisen, they’re given up.”

20

“Mendicants, I do not see a single thing that prevents doubt from arising, or, when it has arisen, gives it up like rational application of mind. When you apply the mind rationally, doubt does not arise, or, if it’s already arisen, it’s given up.”
"""

# Call the function with your text
generate_and_upload(text)
