from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sample customer reviews
text = """
The product quality is excellent and delivery was fast.
Customer service was helpful and polite.
Price is reasonable but packaging could be better.
I love the design and features, very user-friendly.
Will definitely recommend to friends. Excellent value for money.
"""

# Generate Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="plasma").generate(text)

# Plot it
plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("✨ Customer Reviews Word Cloud", fontsize=16, weight="bold")
plt.show()
