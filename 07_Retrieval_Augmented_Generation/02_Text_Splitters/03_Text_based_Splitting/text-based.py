from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""

# Initialize the Splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
)

# Perform the Split
chunks = splitter.split_text(text)

print(chunks)
print()
print("Length of Chunks is: ", len(chunks))
print()
print(chunks[0])
print(chunks[1])
print(chunks[2])
print(chunks[3])
print(chunks[4])