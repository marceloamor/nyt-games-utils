""" -NYT CONNECTIONS SOLVER (Second Attempt)-

NYT Connections is a word grouping game where 16 words must be sorted into 4 groups of 4 words each.
Each group has a common theme or connection, and the difficulty ranges from relatively obvious (yellow) 
to quite obscure (purple).

This solver takes a different approach from the first attempt:
1. Instead of just using semantic similarity (which often fails for word games with puns or cultural references)
2. We'll use a combination of methods, including:
   - Semantic similarity (as a baseline)
   - Common category detection (using WordNet)
   - Common pattern detection (prefixes, suffixes, word structure)
   - Named entity recognition (for brands, places, etc.)
   - External knowledge graphs and databases
   - Thesaurus and dictionary definitions
   
This approach should better handle the variety of connection types that appear in the game.
"""

import requests
import numpy as np
from datetime import datetime
import itertools
from collections import defaultdict, Counter
import re
import json
import os
import time

# Optional imports - we'll try to use if available but gracefully degrade if not
try:
    import nltk
    from nltk.corpus import wordnet
    WORDNET_AVAILABLE = True
    # Download necessary NLTK data if not already present
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
except ImportError:
    WORDNET_AVAILABLE = False
    print("NLTK WordNet not available. Some features will be disabled.")

try:
    import spacy
    # Try to load a spaCy model, starting with larger ones and falling back to smaller ones
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If no model is installed, suggest installation
            print("No spaCy model found. For better results, install one with:")
            print("python -m spacy download en_core_web_sm")
            nlp = None
    SPACY_AVAILABLE = nlp is not None
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    print("spaCy not available. Some features will be disabled.")


class ConnectionsSolver:
    def __init__(self, api_url=None, debug=True):
        """
        Initialize the ConnectionsSolver.
        
        Args:
            api_url: Optional custom API URL for fetching the puzzles
            debug: Whether to print debug information
        """
        self.debug = debug
        self.api_url = api_url or "https://www.nytimes.com/svc/connections/v1/{date}.json"
        self.todays_words = []
        self.word_groups = []
        self.word_vectors = {}
        self.connection_candidates = []
        self.difficulty_colors = ["yellow", "green", "blue", "purple"]  # Easiest to hardest
        
        # Cache for API responses and calculated similarities
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.similarity_cache = {}
        
        # Knowledge bases
        self.word_categories = defaultdict(list)
        self.load_knowledge_bases()

    def load_knowledge_bases(self):
        """Load various knowledge bases and reference data."""
        # For now, we'll just use WordNet if available
        if WORDNET_AVAILABLE:
            self.print_debug("WordNet is available for use")
            
        # Initialize an empty dictionary for detected categories
        self.word_categories = defaultdict(list)

    def detect_categories(self):
        """Detect potential categories dynamically from the current words."""
        self.print_debug("Detecting potential categories")
        
        # Reset categories
        self.word_categories = defaultdict(list)
        
        # 1. Analyze word characteristics
        capitalized_words = []
        all_caps_words = []
        hyphenated_words = []
        
        for word in self.todays_words:
            # Store lowercase version for comparison
            word_lower = word.lower()
            
            # Check formatting characteristics
            if word.isupper() and len(word) > 1:  # Avoid single letters
                all_caps_words.append(word)
            if "-" in word:
                hyphenated_words.append(word)
            if word[0].isupper() and not word.isupper():
                capitalized_words.append(word)
        
        # Add any significant patterns found
        if len(all_caps_words) >= 3:
            self.word_categories["all_caps"] = all_caps_words
            self.print_debug(f"Found all caps words: {all_caps_words}")
            
        if len(hyphenated_words) >= 3:
            self.word_categories["hyphenated"] = hyphenated_words
            self.print_debug(f"Found hyphenated words: {hyphenated_words}")
            
        if len(capitalized_words) >= 3:
            self.word_categories["capitalized"] = capitalized_words
            self.print_debug(f"Found capitalized words: {capitalized_words}")
        
        # 2. If WordNet is available, find common categories
        if WORDNET_AVAILABLE:
            self.detect_wordnet_categories()
            
        # 3. Detect common prefixes and suffixes
        self.detect_affixes()
            
        # 4. Detect semantic clusters using spaCy
        if SPACY_AVAILABLE and nlp is not None:
            self.detect_semantic_clusters()
            
        return self.word_categories
        
    def detect_wordnet_categories(self):
        """Detect common WordNet categories among the words."""
        # Get hypernyms for each word
        word_hypernyms = {}
        for word in self.todays_words:
            # Handle multi-word phrases
            if " " in word:
                parts = word.lower().split()
                word_hypernyms[word] = []
                
                # Try each significant word in the phrase
                for part in parts:
                    if len(part) > 3:  # Skip small words like "the", "of", etc.
                        synsets = wordnet.synsets(part)
                        if synsets:
                            word_hypernyms[word].extend(synsets[:2])  # Take top 2 senses
            else:
                synsets = wordnet.synsets(word.lower())
                if synsets:
                    word_hypernyms[word] = synsets[:2]  # Take top 2 senses
                else:
                    word_hypernyms[word] = []
        
        # Find common hypernyms
        hypernym_counter = Counter()
        word_to_hypernym = defaultdict(list)
        
        for word, synsets in word_hypernyms.items():
            for synset in synsets:
                for hypernym in synset.hypernyms():
                    hypernym_name = hypernym.name().split('.')[0].replace('_', ' ')
                    hypernym_counter[hypernym_name] += 1
                    word_to_hypernym[hypernym_name].append(word)
        
        # Keep hypernyms that appear at least 3 times
        for hypernym, count in hypernym_counter.items():
            if count >= 3:
                words = word_to_hypernym[hypernym]
                self.word_categories[f"category_{hypernym}"] = words
                self.print_debug(f"Found category '{hypernym}': {words}")
                
    def detect_affixes(self):
        """Detect common prefixes and suffixes."""
        # Check for common prefixes
        prefix_counter = Counter()
        word_to_prefix = defaultdict(list)
        
        for word in self.todays_words:
            word_lower = word.lower()
            # Check prefixes of length 2-4
            for prefix_len in range(2, 5):
                if len(word_lower) >= prefix_len:
                    prefix = word_lower[:prefix_len]
                    prefix_counter[prefix] += 1
                    word_to_prefix[prefix].append(word)
        
        # Keep prefixes that appear at least 3 times
        for prefix, count in prefix_counter.items():
            if count >= 3:
                words = word_to_prefix[prefix]
                self.word_categories[f"prefix_{prefix}"] = words
                self.print_debug(f"Found prefix '{prefix}': {words}")
                
        # Check for common suffixes
        suffix_counter = Counter()
        word_to_suffix = defaultdict(list)
        
        for word in self.todays_words:
            word_lower = word.lower()
            # Check suffixes of length 2-4
            for suffix_len in range(2, 5):
                if len(word_lower) >= suffix_len:
                    suffix = word_lower[-suffix_len:]
                    suffix_counter[suffix] += 1
                    word_to_suffix[suffix].append(word)
        
        # Keep suffixes that appear at least 3 times
        for suffix, count in suffix_counter.items():
            if count >= 3:
                words = word_to_suffix[suffix]
                self.word_categories[f"suffix_{suffix}"] = words
                self.print_debug(f"Found suffix '{suffix}': {words}")
                
    def detect_semantic_clusters(self):
        """Detect semantic clusters using spaCy word vectors."""
        if not self.word_vectors:
            self.calculate_word_vectors()
            
        # Skip if we don't have word vectors
        if not self.word_vectors:
            return
            
        # Create a similarity matrix
        words = list(self.word_vectors.keys())
        n = len(words)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    # Calculate cosine similarity
                    vec1 = self.word_vectors[words[i]]
                    vec2 = self.word_vectors[words[j]]
                    
                    # Avoid division by zero
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    
                    if norm1 > 0 and norm2 > 0:
                        sim = np.dot(vec1, vec2) / (norm1 * norm2)
                    else:
                        sim = 0
                        
                    similarity_matrix[i][j] = sim
                    similarity_matrix[j][i] = sim
        
        # Try clustering with different numbers of clusters
        best_silhouette = -1
        best_labels = None
        best_n_clusters = 0
        
        # Try 3-5 clusters
        for n_clusters in range(3, 6):
            if n_clusters >= n:
                continue
                
            try:
                # Apply K-means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(similarity_matrix)
                
                # Skip silhouette calculation for now to keep it simple
                best_labels = labels
                best_n_clusters = n_clusters
                break
            except:
                continue
        
        # If we found clusters, add them to categories
        if best_labels is not None:
            clusters = defaultdict(list)
            for i, label in enumerate(best_labels):
                clusters[label].append(words[i])
            
            for label, cluster_words in clusters.items():
                if len(cluster_words) >= 3:
                    self.word_categories[f"semantic_cluster_{label}"] = cluster_words
                    self.print_debug(f"Found semantic cluster {label}: {cluster_words}")

    def print_debug(self, message):
        """Print debug information if debug mode is enabled."""
        if self.debug:
            print(f"[DEBUG] {message}")

    def print_separator(self):
        """Print a separator line for cleaner output."""
        print("\n" + "=" * 50)

    def fetch_todays_puzzle(self, date=None):
        """Fetch today's NYT Connections puzzle."""
        date = date or datetime.today().strftime("%Y-%m-%d")
        cache_file = os.path.join(self.cache_dir, f"connections_{date}.json")
        
        # Try to load from cache first
        if os.path.exists(cache_file):
            self.print_debug(f"Loading puzzle from cache for {date}")
            with open(cache_file, 'r') as f:
                data = json.load(f)
                self.parse_puzzle_data(data)
                return True
        
        # Otherwise fetch from the API
        try:
            url = self.api_url.format(date=date)
            self.print_debug(f"Fetching puzzle from {url}")
            
            response = requests.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            })
            response.raise_for_status()
            data = response.json()
            
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            self.parse_puzzle_data(data)
            return True
            
        except Exception as e:
            self.print_debug(f"Error fetching puzzle: {e}")
            return False

    def parse_puzzle_data(self, data):
        """Parse the puzzle data from the API response."""
        # Extract the 16 words and shuffle them
        try:
            starting_groups = data.get("startingGroups", [])
            if not starting_groups:
                self.print_debug("No starting groups found in puzzle data")
                return False
                
            self.todays_words = [word for group in starting_groups for word in group]
            self.print_debug(f"Today's words: {self.todays_words}")
            
            # If answers are available, store them for validation
            if "groups" in data:
                self.correct_groups = data["groups"]
                self.print_debug("Answers are available for validation")
            
            # Calculate word vectors if spaCy is available
            if SPACY_AVAILABLE:
                self.calculate_word_vectors()
            
            return True
        except Exception as e:
            self.print_debug(f"Error parsing puzzle data: {e}")
            return False

    def calculate_word_vectors(self):
        """Calculate word vectors for semantic similarity if spaCy is available."""
        if not SPACY_AVAILABLE or nlp is None:
            return
            
        self.print_debug("Calculating word vectors")
        
        # Process all words at once for efficiency
        docs = list(nlp.pipe(self.todays_words))
        
        # Store the vectors for each word
        for word, doc in zip(self.todays_words, docs):
            self.word_vectors[word] = doc.vector

    def calculate_similarity(self, word1, word2):
        """Calculate the similarity between two words."""
        cache_key = tuple(sorted([word1, word2]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        similarity = 0
        
        # 1. Use spaCy vector similarity if available
        if SPACY_AVAILABLE and nlp is not None and word1 in self.word_vectors and word2 in self.word_vectors:
            vec1 = self.word_vectors[word1]
            vec2 = self.word_vectors[word2]
            # Compute cosine similarity with safety check for zero vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 0 and norm2 > 0:  # Avoid division by zero
                similarity += np.dot(vec1, vec2) / (norm1 * norm2)
        
        # 2. Use WordNet similarity if available
        if WORDNET_AVAILABLE:
            synsets1 = wordnet.synsets(word1)
            synsets2 = wordnet.synsets(word2)
            
            if synsets1 and synsets2:
                # Find the maximum similarity between any two synsets
                max_sim = 0
                for s1 in synsets1[:3]:  # Limit to first 3 synsets for performance
                    for s2 in synsets2[:3]:
                        try:
                            # Try different similarity measures
                            path_sim = s1.path_similarity(s2) or 0
                            wup_sim = s1.wup_similarity(s2) or 0
                            # Use the maximum of these similarities
                            max_sim = max(max_sim, path_sim, wup_sim)
                        except:
                            pass
                
                similarity += max_sim
        
        # 3. Check for common prefixes/suffixes (pattern matching)
        prefix_len = 0
        for i in range(min(len(word1), len(word2))):
            if word1[i].lower() == word2[i].lower():  # Case insensitive comparison
                prefix_len += 1
            else:
                break
        
        suffix_len = 0
        for i in range(min(len(word1), len(word2))):
            if word1[-(i+1)].lower() == word2[-(i+1)].lower():  # Case insensitive
                suffix_len += 1
            else:
                break
                
        # Add pattern similarity component (normalized by word length)
        pattern_sim = (prefix_len + suffix_len) / (len(word1) + len(word2))
        similarity += pattern_sim
        
        # 4. Check for substring relationship
        if word1.lower() in word2.lower() or word2.lower() in word1.lower():
            similarity += 0.5
            
        # 5. Check for shared words in multi-word phrases
        word1_parts = set(word1.lower().split())
        word2_parts = set(word2.lower().split())
        common_parts = word1_parts.intersection(word2_parts)
        if common_parts:
            similarity += 0.3 * len(common_parts) / max(len(word1_parts), len(word2_parts))
            
        # Normalize and cache the result
        normalized_similarity = similarity / 3.5  # Adjusted normalization factor
        self.similarity_cache[cache_key] = normalized_similarity
        
        return normalized_similarity

    def find_structural_patterns(self):
        """Look for structural patterns among words."""
        self.print_debug("Analyzing structural patterns")
        
        patterns = {}
        
        # Check for words with common prefixes
        prefixes = defaultdict(list)
        for word in self.todays_words:
            for prefix_len in range(1, 5):  # Check prefixes of length 1-4
                if len(word) >= prefix_len:
                    prefix = word[:prefix_len]
                    prefixes[prefix].append(word)
        
        # Keep only prefixes that appear in at least 3 words
        common_prefixes = {prefix: words for prefix, words in prefixes.items() if len(words) >= 3}
        patterns["prefixes"] = common_prefixes
        
        # Check for words with common suffixes
        suffixes = defaultdict(list)
        for word in self.todays_words:
            for suffix_len in range(1, 5):  # Check suffixes of length 1-4
                if len(word) >= suffix_len:
                    suffix = word[-suffix_len:]
                    suffixes[suffix].append(word)
        
        # Keep only suffixes that appear in at least 3 words
        common_suffixes = {suffix: words for suffix, words in suffixes.items() if len(words) >= 3}
        patterns["suffixes"] = common_suffixes
        
        # Check for word length patterns
        word_lengths = defaultdict(list)
        for word in self.todays_words:
            word_lengths[len(word)].append(word)
        
        patterns["lengths"] = word_lengths
        
        # Check for other patterns like all caps, hyphenated words, etc.
        special_patterns = defaultdict(list)
        for word in self.todays_words:
            if word.isupper():
                special_patterns["all_caps"].append(word)
            if "-" in word:
                special_patterns["hyphenated"].append(word)
            if word[0].isupper():
                special_patterns["capitalized"].append(word)
        
        patterns["special"] = special_patterns
        
        self.print_debug(f"Found structural patterns: {patterns}")
        return patterns

    def find_thematic_connections(self):
        """Find thematic connections between words based on detected categories."""
        self.print_debug("Looking for thematic connections")
        
        # First detect categories
        self.detect_categories()
        
        thematic_candidates = []
        
        # Check for words in our detected categories
        for category_name, category_words in self.word_categories.items():
            # If we found at least 3 matches, create a candidate group
            if len(category_words) >= 3:
                self.print_debug(f"Found {len(category_words)} words in category '{category_name}': {category_words}")
                
                # If we have exactly 4, use them directly
                if len(category_words) == 4:
                    thematic_candidates.append({
                        "group": tuple(category_words),
                        "score": 0.9,  # High score for direct category matches
                        "method": "thematic",
                        "reason": f"Words sharing {category_name.replace('_', ' ')}"
                    })
                # If we have more than 4, take the first 4
                elif len(category_words) > 4:
                    thematic_candidates.append({
                        "group": tuple(category_words[:4]),
                        "score": 0.85,
                        "method": "thematic",
                        "reason": f"Words sharing {category_name.replace('_', ' ')}"
                    })
                # If we have 3, try to find a 4th that's somewhat related
                else:
                    # Calculate similarity of each remaining word to this group
                    remaining_words = [w for w in self.todays_words if w not in category_words]
                    best_similarity = -1
                    best_word = None
                    
                    for word in remaining_words:
                        total_similarity = sum(self.calculate_similarity(word, match) for match in category_words)
                        avg_similarity = total_similarity / len(category_words)
                        
                        if avg_similarity > best_similarity:
                            best_similarity = avg_similarity
                            best_word = word
                    
                    if best_word:
                        matching_words = list(category_words)
                        matching_words.append(best_word)
                        thematic_candidates.append({
                            "group": tuple(matching_words),
                            "score": 0.8,
                            "method": "thematic",
                            "reason": f"Words sharing {category_name.replace('_', ' ')} (with one possible outlier)"
                        })
        
        return thematic_candidates

    def find_character_groups(self):
        """Find groups of fictional characters or people."""
        # This function is no longer needed as we detect categories dynamically
        return []

    def find_category_candidates(self):
        """Find potential categories for the words."""
        candidates = []
        
        # 1. First check for thematic connections from our knowledge bases
        thematic_candidates = self.find_thematic_connections()
        candidates.extend(thematic_candidates)
        
        # 2. Look for groups based on high intra-group similarity
        self.print_debug("Finding groups based on similarity")
        
        # Generate all possible groupings of 4 words
        all_groups = list(itertools.combinations(self.todays_words, 4))
        
        # Calculate the average similarity within each group
        group_similarities = []
        for group in all_groups:
            similarities = []
            for word1, word2 in itertools.combinations(group, 2):
                similarities.append(self.calculate_similarity(word1, word2))
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            group_similarities.append((group, avg_similarity))
        
        # Sort groups by similarity (highest first)
        group_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top N groups as candidates
        top_similarity_groups = group_similarities[:20]
        for group, similarity in top_similarity_groups:
            candidates.append({
                "group": group,
                "score": similarity,
                "method": "similarity",
                "reason": f"High semantic similarity ({similarity:.2f})"
            })
        
        # 3. Look for structural patterns
        patterns = self.find_structural_patterns()
        
        # Check prefixes
        for prefix, words in patterns["prefixes"].items():
            if len(words) >= 4:
                group = tuple(words[:4])
                candidates.append({
                    "group": group,
                    "score": 0.8,  # Assign a high score to pattern matches
                    "method": "prefix",
                    "reason": f"Common prefix: '{prefix}'"
                })
        
        # Check suffixes
        for suffix, words in patterns["suffixes"].items():
            if len(words) >= 4:
                group = tuple(words[:4])
                candidates.append({
                    "group": group,
                    "score": 0.8,
                    "method": "suffix",
                    "reason": f"Common suffix: '{suffix}'"
                })
        
        # Check word lengths
        for length, words in patterns["lengths"].items():
            if len(words) >= 4:
                group = tuple(words[:4])
                candidates.append({
                    "group": group,
                    "score": 0.7,
                    "method": "length",
                    "reason": f"Same length: {length} characters"
                })
        
        # Check special patterns
        for pattern_name, words in patterns["special"].items():
            if len(words) >= 4:
                group = tuple(words[:4])
                candidates.append({
                    "group": group,
                    "score": 0.75,
                    "method": "special",
                    "reason": f"Special pattern: {pattern_name}"
                })
        
        # 4. If WordNet is available, look for common hypernyms
        if WORDNET_AVAILABLE:
            self.print_debug("Finding groups based on WordNet categories")
            
            # Get all synsets for each word
            word_synsets = {}
            for word in self.todays_words:
                # For multi-word phrases, try both the whole phrase and individual words
                if " " in word:
                    # Try the whole phrase
                    whole_synsets = wordnet.synsets(word)
                    
                    # Try individual words if the whole phrase doesn't work well
                    if not whole_synsets:
                        parts = word.split()
                        for part in parts:
                            if len(part) > 3:  # Skip short words like "the", "of", etc.
                                part_synsets = wordnet.synsets(part)
                                if part_synsets:
                                    if word not in word_synsets:
                                        word_synsets[word] = []
                                    word_synsets[word].extend(part_synsets[:2])  # Add top 2 synsets
                    else:
                        word_synsets[word] = whole_synsets
                else:
                    word_synsets[word] = wordnet.synsets(word)
            
            # Find words that share common hypernyms
            hypernym_groups = defaultdict(list)
            
            for word, synsets in word_synsets.items():
                if not synsets:
                    continue
                
                # Get hypernyms for the first few senses of the word
                for synset in synsets[:2]:  # Consider only first 2 senses for performance
                    for hypernym in synset.hypernyms():
                        hypernym_name = hypernym.name().split('.')[0].replace('_', ' ')
                        hypernym_groups[hypernym_name].append(word)
            
            # Find hypernyms that group at least 3 words
            for hypernym, words in hypernym_groups.items():
                if len(words) >= 3:
                    group = tuple(words)
                    candidates.append({
                        "group": group,
                        "score": 0.6,
                        "method": "wordnet",
                        "reason": f"Common category in WordNet: {hypernym}"
                    })
        
        # Remove duplicate groups and groups with fewer than 4 words
        unique_candidates = []
        seen_groups = set()
        
        for candidate in candidates:
            group = candidate["group"]
            if len(group) != 4:
                continue
                
            # Sort the group to ensure uniqueness check works
            sorted_group = tuple(sorted(group))
            if sorted_group not in seen_groups:
                seen_groups.add(sorted_group)
                unique_candidates.append(candidate)
        
        # Sort by score
        unique_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        self.connection_candidates = unique_candidates
        return unique_candidates

    def find_optimal_grouping(self):
        """Find the optimal grouping of all 16 words into 4 groups."""
        self.print_debug("Finding optimal grouping")
        
        # Find candidate groups
        candidates = self.find_category_candidates()
        
        if not candidates:
            self.print_debug("No candidate groups found")
            return None
        
        self.print_debug(f"Found {len(candidates)} candidate groups")
        
        # Print top 5 candidates for debugging
        for i, candidate in enumerate(candidates[:5]):
            self.print_debug(f"Candidate {i+1}: {candidate['group']} - {candidate['reason']} (score: {candidate['score']:.2f})")
        
        # We need to select 4 groups that contain all 16 words with no overlaps
        best_solution = None
        best_score = -1
        
        # Try different combinations of top candidate groups
        # Use more candidates and try different strategies to find a solution
        num_candidates_to_try = min(50, len(candidates))  # Try up to 50 candidates
        
        self.print_debug(f"Trying combinations of top {num_candidates_to_try} candidates")
        
        # Try to find a solution with high-scoring thematic groups first
        thematic_candidates = [c for c in candidates[:num_candidates_to_try] if c["method"] in ("thematic", "character")]
        
        if len(thematic_candidates) >= 2:
            self.print_debug(f"Trying combinations with {len(thematic_candidates)} thematic candidates")
            
            # Try combinations that include at least one thematic group
            for combo in itertools.combinations(candidates[:num_candidates_to_try], 4):
                # Ensure we have at least one thematic group
                if not any(c["method"] in ("thematic", "character") for c in combo):
                    continue
                
                groups = [c["group"] for c in combo]
                words = [word for group in groups for word in group]
                
                # Check if all words are different and we have exactly 16 words
                if len(set(words)) == 16 and len(words) == 16:
                    # Calculate combined score
                    score = sum(c["score"] for c in combo)
                    
                    if score > best_score:
                        best_score = score
                        best_solution = combo
        
        # If we haven't found a solution, try general combinations
        if not best_solution:
            self.print_debug("Trying general combinations")
            
            for combo in itertools.combinations(candidates[:num_candidates_to_try], 4):
                groups = [c["group"] for c in combo]
                words = [word for group in groups for word in group]
                
                # Check if all words are different and we have exactly 16 words
                if len(set(words)) == 16 and len(words) == 16:
                    # Calculate combined score
                    score = sum(c["score"] for c in combo)
                    
                    if score > best_score:
                        best_score = score
                        best_solution = combo
        
        if best_solution:
            self.print_debug(f"Found optimal grouping with score {best_score}")
            for group_info in best_solution:
                self.print_debug(f"Group: {group_info['group']}")
                self.print_debug(f"Reason: {group_info['reason']}")
            
            return best_solution
        else:
            self.print_debug("Could not find an optimal grouping")
            
            # Fallback: create groups based on the best individual candidates we have
            # This won't be perfect but at least gives a result
            self.print_debug("Attempting fallback grouping")
            
            used_words = set()
            fallback_groups = []
            
            for candidate in candidates:
                group = candidate["group"]
                
                # Skip if any words in this group are already used
                if any(word in used_words for word in group):
                    continue
                
                # Add this group
                fallback_groups.append(candidate)
                for word in group:
                    used_words.add(word)
                
                # Stop if we have 4 groups
                if len(fallback_groups) == 4:
                    break
            
            # Check if we found 4 groups that don't overlap
            if len(fallback_groups) == 4:
                self.print_debug("Created fallback grouping")
                return fallback_groups
            
            return None

    def solve(self):
        """Main solving function."""
        # Fetch today's puzzle
        if not self.fetch_todays_puzzle():
            print("Failed to fetch today's puzzle.")
            return
        
        # Find candidates
        candidates = self.find_category_candidates()
        
        if not candidates:
            print("No candidate groups found.")
            return
            
        # First, try to find a complete solution
        solution = self.find_optimal_grouping()
        
        if solution:
            print("\n===== CONNECTIONS SOLVER RESULTS =====")
            print("COMPLETE SOLUTION FOUND:")
            for i, group_info in enumerate(solution):
                color = self.difficulty_colors[i % 4]
                group = group_info["group"]
                reason = group_info["reason"]
                
                print(f"\n{color.upper()} GROUP:")
                for word in group:
                    print(f"  {word}")
                print(f"Connection: {reason}")
            
            print("\n=====================================")
        else:
            # If we can't find a complete solution, present our best group suggestions one at a time
            print("\n===== CONNECTIONS SOLVER SUGGESTIONS =====")
            print("No complete solution found. Here are the most promising individual groups:")
            
            # Get the best group for each method type to ensure diversity in suggestions
            best_by_method = {}
            for candidate in candidates:
                method = candidate["method"]
                if method not in best_by_method or candidate["score"] > best_by_method[method]["score"]:
                    best_by_method[method] = candidate
            
            # Sort methods by their best score
            sorted_methods = sorted(best_by_method.keys(), 
                                   key=lambda m: best_by_method[m]["score"], 
                                   reverse=True)
            
            # Print top 3 best groups (if we have that many)
            print("\nTRY THIS GROUP FIRST:")
            self.print_group_suggestion(best_by_method[sorted_methods[0]], 1)
            
            if len(sorted_methods) > 1:
                print("\nOTHER STRONG CANDIDATES:")
                for i, method in enumerate(sorted_methods[1:4]):  # Up to 3 more suggestions
                    self.print_group_suggestion(best_by_method[method], i+2)
            
            # Suggest an iterative approach
            print("\nSUGGESTED APPROACH:")
            print("1. Try the first group in the game")
            print("2. If correct, run the solver again with the remaining words")
            print("3. If incorrect, try the next suggestion or provide feedback")
            
            print("\n=====================================")
    
    def print_group_suggestion(self, group_info, rank):
        """Print a formatted group suggestion."""
        confidence = group_info["score"] * 100
        
        # Highlight the suggestion with a border for visibility
        print("\n┌" + "─" * 48 + "┐")
        print(f"│ SUGGESTION #{rank} (Confidence: {confidence:.1f}%):" + " " * (15 - len(str(rank))) + "│")
        print("├" + "─" * 48 + "┤")
        
        # Print the group with numbers for reference
        for i, word in enumerate(group_info["group"]):
            print(f"│ {i+1}. {word}" + " " * (45 - len(word)) + "│")
        
        # Print the reason
        print("├" + "─" * 48 + "┤")
        reason = group_info["reason"]
        # Wrap long reasons
        if len(reason) > 46:
            parts = [reason[i:i+46] for i in range(0, len(reason), 46)]
            print(f"│ {parts[0]}" + " " * (46 - len(parts[0])) + "│")
            for part in parts[1:]:
                print(f"│ {part}" + " " * (46 - len(part)) + "│")
        else:
            print(f"│ {reason}" + " " * (46 - len(reason)) + "│")
        
        # Print a confidence assessment
        confidence_msg = ""
        if confidence > 90:
            confidence_msg = "Very high confidence"
        elif confidence > 80:
            confidence_msg = "High confidence"
        elif confidence > 70:
            confidence_msg = "Moderate confidence"
        else:
            confidence_msg = "Lower confidence"
            
        print(f"│ {confidence_msg}" + " " * (46 - len(confidence_msg)) + "│")
        print("└" + "─" * 48 + "┘")

    def solve_iterative(self, known_groups=None, remaining_words=None):
        """Solve the puzzle iteratively, using feedback from previous attempts.
        
        Args:
            known_groups: List of groups already found to be correct
            remaining_words: List of words still to be grouped, if provided
        """
        # Initialize
        self.known_groups = known_groups or []
        
        # If remaining words are provided, use them instead of fetching the puzzle
        if remaining_words:
            self.todays_words = remaining_words
            self.print_debug(f"Using provided remaining words: {self.todays_words}")
            
            # Recalculate word vectors if spaCy is available
            if SPACY_AVAILABLE:
                self.calculate_word_vectors()
        else:
            # Fetch today's puzzle normally
            if not self.fetch_todays_puzzle():
                print("Failed to fetch today's puzzle.")
                return
        
        # Find the best single group
        best_group = self.find_best_single_group()
        
        if best_group:
            print("\n===== NEXT GROUP SUGGESTION =====")
            print(f"Words remaining: {len(self.todays_words)}")
            
            self.print_group_suggestion(best_group, 1)
            
            # Get a few alternate suggestions
            alternates = self.find_alternate_groups(best_group["group"])
            if alternates:
                print("\nALTERNATE SUGGESTIONS:")
                for i, group_info in enumerate(alternates[:2]):  # Show up to 2 alternates
                    self.print_group_suggestion(group_info, i+2)
            
            print("\nAfter trying this group, run the solver again with the remaining words.")
            print("=====================================")
        else:
            print("No more group suggestions available.")
    
    def find_best_single_group(self):
        """Find the single best group from the current words."""
        candidates = self.find_category_candidates()
        
        if not candidates:
            return None
            
        # First, try to find groups from our knowledge bases as they tend to be more reliable
        thematic_candidates = [c for c in candidates if c["method"] in ("thematic", "character")]
        if thematic_candidates:
            return thematic_candidates[0]
        
        # Next, try pattern-based groups
        pattern_candidates = [c for c in candidates if c["method"] in ("prefix", "suffix", "length", "special")]
        if pattern_candidates:
            return pattern_candidates[0]
        
        # Finally, fall back to similarity-based groups
        return candidates[0] if candidates else None
    
    def find_alternate_groups(self, primary_group):
        """Find alternate groups that don't overlap with the primary group."""
        candidates = self.find_category_candidates()
        
        if not candidates:
            return []
            
        # Filter out candidates that share words with the primary group
        primary_words = set(primary_group)
        alternates = [c for c in candidates if not any(word in primary_words for word in c["group"])]
        
        # Ensure we have different methods for diversity in suggestions
        unique_method_alternates = []
        seen_methods = set()
        
        for candidate in alternates:
            if candidate["method"] not in seen_methods:
                unique_method_alternates.append(candidate)
                seen_methods.add(candidate["method"])
                
                # Get at most 3 alternates with different methods
                if len(unique_method_alternates) >= 3:
                    break
        
        return unique_method_alternates

    def solve_interactive(self):
        """Interactive mode that guides the user through solving the puzzle."""
        # Fetch today's puzzle
        if not self.fetch_todays_puzzle():
            print("Failed to fetch today's puzzle.")
            return
            
        self.print_separator()
        print("CONNECTIONS SOLVER - INTERACTIVE MODE")
        print("This mode will guide you through solving the puzzle step by step.")
        print("You'll get suggestions, try them in the game, and provide feedback.")
        
        # Keep track of correct groups and remaining words
        correct_groups = []
        remaining_words = self.todays_words.copy()
        attempts_left = 4
        
        while remaining_words and attempts_left > 0:
            self.print_separator()
            print(f"ROUND {5-attempts_left} | Words remaining: {len(remaining_words)} | Attempts left: {attempts_left}")
            
            # Show remaining words in a grid format for better visibility
            print("\nREMAINING WORDS:")
            words_per_row = 4
            for i in range(0, len(remaining_words), words_per_row):
                row_words = remaining_words[i:i+words_per_row]
                print("  ".join(f"{j+1+i}. {word}" for j, word in enumerate(row_words)))
                
            # Find the best group suggestion
            self.todays_words = remaining_words
            if SPACY_AVAILABLE:
                self.calculate_word_vectors()
                
            best_group = self.find_best_single_group()
            
            if not best_group:
                print("No more suggestions available.")
                break
                
            # Show the suggestion
            self.print_group_suggestion(best_group, 1)
            
            # Get alternates
            alternates = self.find_alternate_groups(best_group["group"])
            
            # Ask user to try the suggestion and provide feedback
            print("\nACTIONS:")
            print("1. Group is correct! (The game accepted it)")
            print("2. Group is one away (The game said 'one away')")
            print("3. Group is wrong (The game rejected it)")
            print("4. Show alternate suggestions")
            print("5. Enter my own group")
            print("q. Quit interactive mode")
            
            choice = input("\nEnter your choice (1-5, q): ").strip().lower()
            
            if choice == 'q':
                print("Exiting interactive mode.")
                break
                
            if choice == '4':
                # User wants to see more suggestions
                if alternates:
                    print("\nALTERNATE SUGGESTIONS:")
                    for i, group_info in enumerate(alternates[:2]):
                        self.print_group_suggestion(group_info, i+2)
                else:
                    print("\nNo alternate suggestions available.")
                continue
                
            if choice == '5':
                # User wants to enter their own group
                print("\nSelect four words from the remaining words (enter the numbers):")
                print("Example: 1 5 7 12")
                
                try:
                    selections = input("Your selection: ").strip().split()
                    if len(selections) != 4:
                        print("You must select exactly 4 words.")
                        continue
                        
                    indices = [int(s) - 1 for s in selections]
                    custom_group = []
                    
                    for idx in indices:
                        if 0 <= idx < len(remaining_words):
                            custom_group.append(remaining_words[idx])
                        else:
                            print(f"Invalid selection: {idx+1}")
                            custom_group = []
                            break
                    
                    if not custom_group or len(custom_group) != 4:
                        print("Invalid selection. Please try again.")
                        continue
                        
                    # Use the custom group
                    print("\nYour selected group:")
                    for word in custom_group:
                        print(f"  {word}")
                        
                    confirm = input("\nIs this correct? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                        
                    selected_group = tuple(custom_group)
                    
                    # Ask if this group was correct in the game
                    result = input("\nWas this group correct in the game? (y/n): ").strip().lower()
                    if result == 'y':
                        correct_groups.append(selected_group)
                        attempts_left -= 1
                    else:
                        attempts_left -= 1
                        print("Let's try a different approach.")
                        continue
                        
                except ValueError:
                    print("Please enter valid numbers.")
                    continue
            else:
                # User tried one of our suggestions
                if choice == '1' or choice == '2' or choice == '3':
                    attempts_left -= 1
                    
                    if choice == '1':
                        # Group was correct
                        selected_group = best_group["group"]
                        correct_groups.append(selected_group)
                        print("Great! Let's continue with the remaining words.")
                    elif choice == '2':
                        # One away - we need to try all possible combinations
                        print("\nSince the game said 'one away' but doesn't tell you which word,")
                        print("let's try to identify the misplaced word.")
                        
                        # Get the current group
                        current_group = list(best_group["group"])
                        
                        # Get other words not in the group
                        other_words = [w for w in remaining_words if w not in current_group]
                        
                        if not other_words:
                            print("No other words available to swap. This is unusual.")
                            continue
                            
                        print("\nI'll suggest word swaps to try. For each, answer if it worked.")
                        
                        found_correct_group = False
                        
                        # Try swapping each word in the group with each word outside
                        for i, word_to_remove in enumerate(current_group):
                            if found_correct_group:
                                break
                                
                            for j, word_to_add in enumerate(other_words[:min(5, len(other_words))]):
                                # Create a new group with the swap
                                new_group = current_group.copy()
                                new_group[i] = word_to_add
                                
                                print("\nTry this group:")
                                for word in new_group:
                                    print(f"  {word}")
                                print(f"\n(Replaced {word_to_remove} with {word_to_add})")
                                
                                swap_result = input("\nDid this group work? (y/n/skip): ").strip().lower()
                                
                                if swap_result == 'y':
                                    selected_group = tuple(new_group)
                                    correct_groups.append(selected_group)
                                    found_correct_group = True
                                    print("Great! Let's continue with the remaining words.")
                                    break
                                elif swap_result == 'skip':
                                    print("Skipping remaining swaps.")
                                    break
                        
                        if not found_correct_group:
                            print("\nCouldn't find the correct group through swaps.")
                            print("Let's try a different approach.")
                            continue
                        else:
                            # We found the correct group, so set selected_group
                            pass  # selected_group is already set above
                    else:  # choice == '3'
                        # Group was wrong, continue to next suggestion
                        print("Let's try a different approach.")
                        continue
                else:
                    print("Invalid choice. Please enter 1-5 or q.")
                    continue
            
            # Remove the selected group's words from remaining_words
            for word in selected_group:
                if word in remaining_words:
                    remaining_words.remove(word)
            
            # Update the user on progress
            if len(correct_groups) == 4:
                print("\nCongratulations! You've solved the entire puzzle!")
                break
                
            if not remaining_words:
                print("\nAll words have been grouped!")
                break
                
            if attempts_left == 0:
                print("\nYou've used all your attempts. Game over!")
                break
        
        # Show final results
        self.print_separator()
        print("FINAL RESULTS")
        if correct_groups:
            print(f"\nYou found {len(correct_groups)} correct groups:")
            for i, group in enumerate(correct_groups):
                print(f"\nGroup {i+1}:")
                for word in group:
                    print(f"  {word}")
        else:
            print("No correct groups found.")
            
        if remaining_words:
            print(f"\n{len(remaining_words)} words remain ungrouped:")
            for word in remaining_words:
                print(f"  {word}")
        
        print("\nThanks for using the Connections Solver!")
        self.print_separator()


if __name__ == "__main__":
    solver = ConnectionsSolver(debug=True)
    
    # Choose the mode
    print("NYT Connections Solver")
    print("1. Interactive Mode (recommended)")
    print("2. Suggestion Mode")
    print("3. Debug Mode (verbose output)")
    
    mode = input("Select mode (1-3): ").strip()
    
    if mode == '1':
        solver.solve_interactive()
    elif mode == '2':
        solver.solve()
    else:
        # Debug mode with extra output
        solver.debug = True
        solver.solve() 