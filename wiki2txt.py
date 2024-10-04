import wikipediaapi

# Initialize the Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia('CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)')
def get_wikipedia_text(wiki_link):
    # Extract the title from the Wikipedia link
    title = wiki_link.split('/')[-1]

    # Get the page
    page = wiki_wiki.page(title)
    
    # Check if the page exists
    if not page.exists():
        return "The specified Wikipedia page does not exist."
    
    # Return the text content of the page
    return page.text

# Example usage
wiki_link = "https://en.wikipedia.org/wiki/Python_(programming_language)"
text = get_wikipedia_text(wiki_link)
print(text)
