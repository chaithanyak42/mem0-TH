
import asyncio, csv, json, shutil, subprocess, sys, os
from datetime import datetime
from pathlib     import Path
from typing      import Dict, List
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from mem0.configs.embeddings.base import BaseEmbedderConfig


#MODEL_TAG  = "llama3.1:8b-instruct-q4_K_M"      
MODEL_TAG ="llama4:17b-scout-16e-instruct-fp16"
EMBED_TAG  = "nomic-embed-text"                  
USER_ID    = "debug-user"  
AGENT_ID   = "debug-agent"  
TOP_K      = 5
OUT_DIR    = Path("benchmarks"); OUT_DIR.mkdir(exist_ok=True)

home_dir = os.path.expanduser("~")
mem0_dir = os.environ.get("MEM0_DIR") or os.path.join(home_dir, ".mem0")

EMBEDDING_DIMS = 768  

MEMORIES: List[str] = [
    "Favourite colour is blue.",
    "I was born in Hyderabad.",
    "I am allergic to strawberries.",
    "I prefer coffee over tea.",
    "My birthday is on 14 June.",
    "I play badminton every Saturday.",
    "My dog's name is Bruno.",
    "I graduated from IIT Bombay.",
    "My favourite genre is science‑fiction.",
    "My comfort food is ramen.",
    "I speak English, Hindi and basic Spanish.",
    "I wake up at 5:30 a.m. on weekdays.",
    "Python is my go‑to language.",
    "I ran a half‑marathon in 2023.",
    "I write a gratitude journal nightly.",
    "I prefer mountains to beaches.",
    "Lo‑fi beats help me focus.",
    "My EU shoe size is 43.",
    "I'm saving for a trip to Japan.",
    "I meditate ten minutes daily.",
    "I have never eaten sushi.",
    "I wear contact lenses.",
    "My favourite board game is Catan.",
    "I never drink beer.",
    "I am ovo‑vegetarian.",
    "I once broke my left arm.",
    "I always use dark mode.",
    "I'm learning the ukulele.",
    "My laptop runs Arch Linux.",
    "I collect postage stamps.",
    "I bake sourdough bread on weekends.",
    "I commute by bicycle.",
    "I love watching cricket.",
    "I order food extra spicy.",
    "I'm afraid of heights.",
    "I water plants every Wednesday.",
    "My cat is a Bengal.",
    "I reread 'The Alchemist' every year.",
    "I drink three litres of water daily.",
    "I lived in Singapore for six months.",
    "I use the metric system.",
    "I am an only child.",
    "I solve Sudoku before bed.",
    "I drive a Honda Civic.",
    "I never skip breakfast.",
    "The monsoon season is my favourite.",
    "I sit on a standing desk at home.",
    "I volunteer on Fridays at an animal shelter.",
    "I am lactose‑intolerant.",
    "I follow yoga sessions every evening.",
    "I listen to podcasts while commuting.",
    "I collect model aeroplanes.",
    "I have visited 12 countries.",
    "Mango flavour is my weakness.",
    "I can solve a Rubik's cube quickly.",
    "I avoid horror movies.",
    "I wear a Fitbit daily.",
    "I have never fractured a bone.",
    "My middle name is Raj.",
    "I only drink green tea.",
    "Astronomy fascinates me.",
    "I type with the Dvorak layout.",
    "My phone stays on Do Not Disturb.",
    "I rescued a stray kitten named Luna.",
    "I love photographing rain reflections.",
    "I can't swim yet.",
    "Pistachio ice‑cream is my favourite.",
    "I use noise‑cancelling headphones.",
    "I donate blood twice a year.",
    "I prefer handwritten notes.",
    "I never miss a Formula 1 race.",
    "I dislike pineapple on pizza.",
    "I collect fountain pens.",
    "My home server runs Ubuntu.",
    "I studied French for two semesters.",
    "I love watching sunrise over the ocean.",
    "I carry a Kindle when travelling.",
    "I practice calligraphy on Sundays.",
    "I'm allergic to cats.",
    "My first car was a Maruti 800.",
    "A tiny bonsai sits on my desk.",
    "The Himalayas are my favourite hiking spot.",
    "After 4 p.m. I switch to decaf.",
    "I choose sci‑fi over rom‑com.",
    "I run full backups every Monday.",
    "I can juggle three balls.",
    "Olives are my favourite pizza topping.",
    "I played drums in college.",
    "I disable social‑media apps during work hours.",
    "I once played Hamlet on stage.",
    "I visit my grandparents monthly.",
    "I read paperbacks, not e‑books, when flying.",
    "I keep dark chocolate in my desk.",
    "I always carry a metal water bottle.",
    "Spiders scare me.",
    "I grow basil and mint on my balcony.",
    "I track sleep with a smart‑watch.",
    "I adopted a beagle named Oliver.",
    "I read The Economist weekly.",
    "I plan to learn snowboarding next winter."
]

QUERIES = [
    "When I'm shopping for clothes and need to pick a colour that always makes me happy, which shade do I inevitably choose?",
    "During small‑talk someone asks about my birthplace— which Indian tech hub city do I proudly mention?",
    "If a dessert contains certain berries that trigger my allergy, which specific fruit must I avoid to stay safe?",
    "Faced with the choice between the world's most aromatic coffee and a perfectly brewed tea, which beverage do I actually prefer?",
    "On which calendar date do friends need to wish me happy birthday so they don't miss my celebration?",
    "Which sport keeps me active every Saturday morning on the court rather than sleeping in?",
    "What is the name of the canine companion who greets me when I get home?",
    "When asked where I earned my undergraduate engineering degree, which IIT campus do I cite?",
    "In our book‑club, which genre do I always lobby for because I can't resist futuristic adventures?",
    "If I'm feeling down and need comfort food, which steaming bowl from Japanese cuisine do I cook or order?",
    "List the languages I can converse in, including the European one I still speak only a little.",
    "At what early hour does my weekday alarm routinely ring?",
    "Which programming language do I hold dearest for its readability and versatility?",
    "Which long‑distance running milestone did I conquer back in 2023?",
    "What nightly reflective practice do I complete before going to sleep?",
    "Do I favour mountain landscapes or sandy beaches when planning a vacation—and which of the two do I pick?",
    "What genre of background music do I stream to stay in flow while working?",
    "If I'm buying shoes in Europe, what EU size do I usually request?",
    "Next year's big trip destination is on my savings vision‑board; which country is it?",
    "Which daily mindfulness habit occupies exactly ten minutes of my schedule?",
    "Name the iconic Japanese dish I have *never* tasted, despite its popularity.",
    "Instead of glasses, what corrective vision option do I wear?",
    "During a game night I always suggest one particular strategy board game— which is it?",
    "When the waiter asks for my drink order, which alcoholic option will I *always* decline?",
    "Describe my dietary preference that allows eggs but excludes meat.",
    "Which bone did I break as a child, the memory of which still makes me cautious?",
    "When an app offers light or dark mode, which theme do I instinctively enable?",
    "Which cheerful string instrument am I currently learning to strum?",
    "What Linux distribution powers my everyday laptop?",
    "Which nostalgic hobby involving tiny adhesive squares do I pursue?",
    "What artisanal bread—famous for its tangy starter—do I bake on weekends?",
    "What eco‑friendly mode of transport do I use for my daily commute?",
    "Which spectator sport do I most enjoy watching, especially international matches?",
    "When ordering food, what spice level do I unapologetically demand?",
    "Identify the phobia I experience when I'm near edges of tall buildings.",
    "On which day of the week do my houseplants reliably get watered?",
    "What breed of cat—known for its striking coat—shares my home?",
    "Which Paulo Coelho novel have I read no fewer than five times?",
    "Roughly how much water do I aim to drink over a full day?",
    "In which Southeast‑Asian city‑state did I spend half a year living?",
    "Which system of measurement (metric or imperial) do I naturally default to?",
    "What unique family fact applies to my sibling situation?",
    "Every night before sleeping I tackle logic puzzles— which classic number grid is my favourite?",
    "What make and model of car am I currently driving?",
    "Which essential meal of the day do I insist on never skipping?",
    "Among the four seasons, which rain‑soaked one brings me the most joy?",
    "What type of ergonomic furniture do I use when working from home?",
    "On which weekday do I volunteer at the animal shelter?",
    "Which dairy sensitivity affects my diet choices?",
    "What form of exercise routine do I practice on weekday evenings to stay limber?",
    "When commuting, do I prefer listening to traditional radio or another audio format—and which one?",
    "What niche collection of tiny aircraft models do I keep?",
    "How many countries have stamped my passport so far?",
    "Which tropical fruit flavour makes any dessert irresistible to me?",
    "What party trick involves solving a colourful cube in under two minutes?",
    "Which film genre do I avoid because it keeps me up at night?",
    "Which wearable device do I strap on each day to monitor health metrics?",
    "Have I ever suffered a bone fracture in my life— yes or no?",
    "What is my middle name?",
    "Between green tea and black tea, which variant earns my loyalty?",
    "Which scientific field dealing with celestial bodies fascinates me?",
    "Which alternative keyboard layout, famed for efficiency, do I type with?",
    "What notification setting do I keep my phone on throughout the workday?",
    "What animal did I rescue and adopt last year?",
    "Describe the weather scene I adore photographing right after rainfall.",
    "What important aquatic life skill have I *not* yet mastered?",
    "What nut‑based ice‑cream flavour tops my favourites list?",
    "What kind of high‑tech headphones help me focus at work?",
    "How many times per year do I typically donate blood?",
    "Do I prefer jotting notes by hand or typing—and which medium wins?",
    "Which fast‑paced motorsport do I schedule to watch live on weekends?",
    "Which controversial pizza topping do I consistently dislike?",
    "What collectible writing instruments—ink‑fed—do I treasure?",
    "Which operating system powers my home server?",
    "Which foreign language did I study for two semesters?",
    "When chasing dawn views, what natural phenomenon do I enjoy witnessing at the beach?",
    "What e‑reader device accompanies me on travels?",
    "What elegant penmanship art form do I practice on weekends?",
    "Which common household pet allergy affects me?",
    "What model was the very first car I owned?",
    "What living miniature plant sits on my workspace?",
    "Which mountain range is my preferred hiking destination?",
    "After 4 p.m. I switch to a low‑caffeine beverage; which coffee variety is it?",
    "Between sci‑fi and rom‑com films, which genre do I gravitate toward on movie night?",
    "On which day do I routinely run full data backups?",
    "What circus‑style skill involving three objects can I perform?",
    "Which Mediterranean pizza topping is my favourite?",
    "Which percussion instrument did I play during college days?",
    "What strict rule do I follow about social media during work hours?",
    "Which Shakespearean character did I once portray on stage?",
    "How frequently do I visit my grandparents?",
    "When travelling, do I pack paperbacks or an e‑reader—and which do I actually use?",
    "What sweet emergency provision do I keep hidden in my desk?",
    "Which reusable hydration accessory do I always pack when travelling?",
    "What eight‑legged creatures trigger my fear response?",
    "Which two edible plants thrive on my balcony garden?",
    "Which wearable gadget do I rely on to analyse my sleep cycles?",
    "What breed of dog named Oliver did I adopt?",
    "Which weekly publication covering global affairs do I read religiously?",
    "Which winter sport am I planning to learn next season?"
]
assert len(MEMORIES) == len(QUERIES) == 100


def ensure_ollama(tag: str) -> None:
    if shutil.which("ollama") is None:
        sys.exit("Install Ollama first: https://ollama.com")
    models = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    if tag.split(":")[0] not in models:
        print(f"Pulling {tag} …", flush=True)
        subprocess.run(["ollama", "pull", tag], check=True)

async def run_benchmark() -> Dict:
    ensure_ollama(MODEL_TAG)
    ensure_ollama(EMBED_TAG)

    qdrant_data_path = os.path.join(mem0_dir, "qdrant_data")
    history_db_path = os.path.join(mem0_dir, "history.db")
    
    if os.path.exists(qdrant_data_path):
        print(f"🧹 Cleaning up old vector store data...")
        try:
            shutil.rmtree(qdrant_data_path)
            print(f"✓ Removed {qdrant_data_path}")
        except Exception as e:
            print(f"Note: Failed to remove {qdrant_data_path}: {e}")
    
    if os.path.exists(history_db_path):
        try:
            os.remove(history_db_path)
            print(f"✓ Removed {history_db_path}")
        except Exception as e:
            print(f"Note: Failed to remove {history_db_path}: {e}")

    print("setting up with embedding dims:", EMBEDDING_DIMS)
    
    from mem0 import AsyncMemory
    from mem0.llms.configs import LlmConfig
    from mem0.configs.base import MemoryConfig
    from mem0.embeddings.configs import EmbedderConfig
    from mem0.vector_stores.configs import VectorStoreConfig
    from mem0.configs.vector_stores.qdrant import QdrantConfig
    from mem0.vector_stores.qdrant import Qdrant
    from mem0.configs.embeddings.base import BaseEmbedderConfig

    cfg = MemoryConfig(version="v1.1")
    
    cfg.llm = LlmConfig(provider="ollama", config={"model": MODEL_TAG})
    
    cfg.embedder = EmbedderConfig(
        provider="ollama",
        config={
            "model": EMBED_TAG,
            "embedding_dims": EMBEDDING_DIMS,
            "ollama_base_url": None
        }
    )
    
    cfg.vector_store = VectorStoreConfig(
        provider="qdrant",
        config={
            "collection_name": f"{USER_ID}_{AGENT_ID}",
            "embedding_model_dims": EMBEDDING_DIMS,
            "path": qdrant_data_path,
            "on_disk": True
        }
    )
    
    print(f"Config:")
    print(f"  Embedder provider: {cfg.embedder.provider}")
    print(f"  Embedder model: {cfg.embedder.config.get('model')}")
    print(f"  Embedding dims: {cfg.embedder.config.get('embedding_dims')}")
    print(f"  Vector store provider: {cfg.vector_store.provider}")
    print(f"  Vector store collection: {cfg.vector_store.config.collection_name}")
    print(f"  Vector store dims: {cfg.vector_store.config.embedding_model_dims}")
    
    mem = AsyncMemory(cfg)
    
    print("testing embedder dimensions...")
    from mem0.embeddings.ollama import OllamaEmbedding
    
    embedder_config = BaseEmbedderConfig(
        model=EMBED_TAG,
        embedding_dims=EMBEDDING_DIMS,
        ollama_base_url=None
    )
    
    embedder = OllamaEmbedding(embedder_config)
    test_vector = embedder.embed("Test embedding dimensions")
    print(f"✓ Test embedding length: {len(test_vector)} dimensions")
    
    if len(test_vector) != EMBEDDING_DIMS:
        print(f"⚠️ WARNING: Actual embedding dimensions ({len(test_vector)}) don't match configured dimensions ({EMBEDDING_DIMS})")
        print(f"⚠️ Updating embedding_model_dims to {len(test_vector)}")
        cfg.vector_store.config["embedding_model_dims"] = len(test_vector)
        cfg.embedder.config["embedding_dims"] = len(test_vector)
        mem = AsyncMemory(cfg)

    print("adding memories …")
    id_map: Dict[str, str] = {}
    for mem_text in tqdm(MEMORIES, unit="mem"):
        res = await mem.add(mem_text, user_id=USER_ID, agent_id=AGENT_ID)
        id_map[mem_text] = res["results"][0]["id"]

    print("searching …")
    hits, rows = 0, []
    for mem_txt, q in atqdm(list(zip(MEMORIES, QUERIES)), total=len(QUERIES), unit="qry"):
        target_id = id_map[mem_txt]
        res   = await mem.search(q, user_id=USER_ID, agent_id=AGENT_ID, limit=TOP_K)
        topKs = [r["id"] for r in res["results"]]
        hit   = target_id in topKs
        hits += hit
        rows.append(
            {"query": q, "target": mem_txt, "target_id": target_id,
             "hit": int(hit), "topK_ids": json.dumps(topKs),
             "top1_memory": res["results"][0]["memory"] if res["results"] else ""}
        )

    p_at_5 = hits / len(QUERIES)
    summary = {
        "model": MODEL_TAG,
        "embedder": EMBED_TAG,
        "timestamp": datetime.utcnow().isoformat(),
        "num_queries": len(QUERIES),
        "precision_at_5": round(p_at_5, 4),
    }

    raw_csv = OUT_DIR / "retrieval_raw.csv"
    with raw_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    with (OUT_DIR / "retrieval_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return summary

if __name__ == "__main__":
    print(f"⏳  benchmark {MODEL_TAG} with embedder {EMBED_TAG}")
    summary = asyncio.run(run_benchmark())
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))