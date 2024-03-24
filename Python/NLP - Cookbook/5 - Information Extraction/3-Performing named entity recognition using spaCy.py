import spacy


nlp = spacy.load("en_core_web_sm")


article = """iPhone 12: Apple makes jump to 5G
Apple has confirmed its iPhone 12 handsets will be its
first to work on faster 5G networks.
The company has also extended the range to include a new
"Mini" model that has a smaller 5.4in screen.
The US firm bucked a wider industry downturn by
increasing its handset sales over the past year.
But some experts say the new features give Apple its best
opportunity for growth since 2014, when it revamped its
line-up with the iPhone 6.
…
"Networks are going to have to offer eye-wateringly
attractive deals, and the way they're going to do that is
on great tariffs and attractive trade-in deals,"
predicted Ben Wood from the consultancy CCS Insight.
Apple typically unveils its new iPhones in September, but
opted for a later date this year.
It has not said why, but it was widely speculated to be
related to disruption caused by the coronavirus pandemic.
The firm's shares ended the day 2.7% lower.
This has been linked to reports that several Chinese
internet platforms opted not to carry the livestream,
although it was still widely viewed and commented on via
the social media network Sina Weibo."""


doc = nlp(article)

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
