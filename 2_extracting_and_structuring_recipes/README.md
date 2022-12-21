[https://binal.pub/2022/12/extracting-and-structuring-recipes-using-gpt3/](https://binal.pub/2022/12/extracting-and-structuring-recipes-using-gpt3/)

The second experiment I've tried is GPT3 to extract and structure data and have been pretty impressed. The below example took me about an hour to setup, most of it just being iterating on the prompts I'm using as directions to the model.

>An additional bonus I wasn't expecting - this also turned out to be a decent recipe generator. If I input in just a recipe name like Pumpkin Pie it'll generate/hallucinate structured ingredients and instructions.

What this does is:
1. Takes free-text recipe and parse out the ingredients and instructions as JSON
    * For ingredients - if quantity is available parse that out separately
2. Take the ingredients parsed out and group them together by category (like dairy, etc)
    * Add this into the JSON as a separate item

All the parsing and structuring is occuring within GPT3 - no complex logic on the Python side.

![Parsing Example](img/gpt3-parsing-recipe.gif)

>In the past when I wanted to do something like this I would have finetuned/trained a named entity recognition model (NER). To start from scratch this involved a lot of time and effort to get going (mainly for steps 1 and 2). Steps 1, 2, and 3 would also have to be repeated if we wanted to add a new entity we'd like to extract.
>1. Gathering a corpus of training data.
>2. Annotating the data (in NER this is tedious since you're annotating individual words with tags)
>3. Trained a model for token classification - all this means is that we predict a label per individual token instead of an overall label
>4. Wrap this all up to take these token level predictions and generate useful output (combining prediction spans, aligning predictions with the original text, etc)

Here's how I set this up using [LangChain](https://langchain.readthedocs.io/en/latest/) - I'd highly recommend you check it out if you have workflows that are moving past single calls to GPT3/other LLMs.

I won't post the entirely of the code (the full code is linked at the top of this post), the interesting bits are below.

First - the prompts I used as templates, I didn't spend too much time here and just went with straightforward directions with no in-context examples (so this is completely one-shot).

```python
# initial prompt to extract the instructions and ingredients
# for ingredients extract quantity as well 
extract_prompt = PromptTemplate(
    input_variables=["recipe"], 
    template="""{recipe}
    
    Extract the ingredients and instructions from the above, output as json with keys instructions and ingredients. For ingredients extract quantity if available."""
)

# secondary prompt to group extracted ingredients into categories
grouping_prompt = PromptTemplate(
    input_variables=["ingredients"], 
    template="""
    {ingredients}
    
    Group above ingredients into similar categories, output as json with lowercase keys categories:"""
)

```

I then run the full text of a recipe through those prompts (LLMChains below fill in the templated prompts and send them to the LLM for inference):

```python
extract_chain = LLMChain(prompt=extract_prompt, llm=llm)
grouping_chain = LLMChain(prompt=grouping_prompt, llm=llm)

text_result = extract_chain.run(recipe)
json_result = json.loads(text_result)
print(text_result)
```

```
{
  "ingredients": [
    {
      "name": "butter",
      "quantity": "1/2 cup (4 ounces or 115 grams)"
    },
    {
      "name": "garlic",
      "quantity": "1 large head"
    },
    {
      "name": "kosher salt"
    },
    ...
  ],
  "instructions": [
    "Heat oven to 375°F (190°C).",
    ...
  ]
}
```
Then we get the ingredient groupings - note here I'm just dumping the json string straight into thee prompt.
```python
ingredient_group_text = grouping_chain.run(json.dumps(json_result['ingredients']))
ingredient_group_json = json.loads(ingredient_group_text)
print(ingedient_group_text)
```
```
{
    "dairy": [{"name": "butter", "quantity": "1/2 cup (4 ounces or 115 grams)"}],
    "vegetables": [{"name": "garlic", "quantity": "1 large head"}, {"name": "baby spinach", "quantity": "5 ounces (140 grams)"}],
    "grains": [{"name": "thin spaghetti", "quantity": "1 pound (455 grams)"}],
    "seasonings": [{"name": "kosher salt"}, {"name": "freshly ground black pepper"}],
    "cheese": [{"name": "Pecorino romano"}]
}
```
And then we just update our initial dict with this enriched data and we're done
```python
json_result['grouped_ingredients'] = ingredient_group_json
print(json_result)
```

```
{
    "ingredients": [
        {
            "name": "butter",
            "quantity": "1/2 cup (4 ounces or 115 grams)"
        },
        {
            "name": "garlic",
            "quantity": "1 large head"
        },
        {
            "name": "kosher salt"
        },
        {
            "name": "baby spinach",
            "quantity": "5 ounces (140 grams)"
        },
        {
            "name": "thin spaghetti",
            "quantity": "1 pound (455 grams)"
        },
        {
            "name": "freshly ground black pepper"
        },
        {
            "name": "Pecorino romano"
        }
    ],
    "instructions": [
        "Heat oven to 375\u00b0F (190\u00b0C).",
        "Arrange the butter slices across the bottom of a small (2-cup) baking dish. Sprinkle with salt: \u00bc teaspoon if using salted butter, and \u00bd teaspoon if unsalted. Place the garlic halves, cut side down, over the butter and salt. Cover the dish tightly with foil, and bake for 35 to 45 minutes, until the garlic is absolutely soft when poked with a knife and golden brown along the cut side. Carefully remove the foil. Empty the garlic cloves into the melted butter. I do this by lifting the peels out of the butter with tongs, allowing most cloves to fall out, and using the tip of a knife to free the cloves that don\u2019t. Scrape any browned bits from the sides of the baking vessel into the butter.",
        "Meanwhile, cook your pasta in well-salted water until 1 to 2 minutes shy of done. Before you drain it, ladle 1 cup pasta water into a cup, and set it aside. Hang on to the pot you cooked the pasta in.",
        "Place the spinach in a blender or food-processor bowl, and pour the garlic butter over it, scraping out any butter left behind. Add another \u00be teaspoon salt and several grinds of black pepper, and/or a couple pinches of red-pepper flakes, and blend the mixture until totally smooth. If it\u2019s not blending, add 1 to 2 tablespoons of reserved pasta water to help it along. Taste for seasoning, and add more if needed.",
        "Pour the spinach sauce into the empty spaghetti pot, and add the drained pasta and a splash of pasta water. Cook over medium-high heat, tossing constantly, for 2 minutes, until the sauce thickens and coats the spaghetti. If the pasta sticks to the bottom of the pot, add more reserved pasta water in splashes to get it moving. Tip the pasta into a serving bowl, finish with more salt and pepper and freshly grated cheese, and hurry\u2014it disappears fast."
    ],
    "grouped_ingredients": {
        "dairy": [
            {
                "name": "butter",
                "quantity": "1/2 cup (4 ounces or 115 grams)"
            }
        ],
        "vegetables": [
            {
                "name": "garlic",
                "quantity": "1 large head"
            },
            {
                "name": "baby spinach",
                "quantity": "5 ounces (140 grams)"
            }
        ],
        "grains": [
            {
                "name": "thin spaghetti",
                "quantity": "1 pound (455 grams)"
            }
        ],
        "seasonings": [
            {
                "name": "kosher salt"
            },
            {
                "name": "freshly ground black pepper"
            }
        ],
        "cheese": [
            {
                "name": "Pecorino romano"
            }
        ]
    }
}
```

Here's a brain dump of things I'd maybe try next (independently or as an extension of this):
* Using the opensource Whisper model + GPT structured parsing to get "video/audio -> X".
   * Video/Audio -> Text -> Embeddings = semantic search
   * Video/Audio -> Text -> Structured Generation from non-text sources?
* Using a secondary model to align images/audio/etc embeddings to OpenAI generated embeddings from their new embedding model https://openai.com/blog/new-and-improved-embedding-model/ 
    * I could then query in the OpenAI embedding space and retrieve non-text data
* Running the above in parallel with chunking - right now for a long recipe it takes ~10-20 seconds to complete. The trade off will be losing global context may result in worse extraction.
* Hooking GPT3 up to a browser/requests
  * https://github.com/nat/natbot
* Generating output with attribution - for my GPT3 output can I attribute it back to a source or reliably verify if it's a hallucination or not.