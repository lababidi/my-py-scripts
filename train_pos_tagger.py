import nltk.tag, itertools, pickle
from nltk.tag import brill

def backoff_tagger(tagged_sents, tagger_classes, backoff=None):
    if not backoff:
        backoff = tagger_classes[0](tagged_sents)
        del tagger_classes[0]
 
    for cls in tagger_classes:
        tagger = cls(tagged_sents, backoff=backoff)
        backoff = tagger
 
    return backoff

word_patterns = [
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
    (r'.*ould$', 'MD'),
    (r'.*ing$', 'VBG'),
    (r'.*ed$', 'VBD'),
    (r'.*ness$', 'NN'),
    (r'.*ment$', 'NN'),
    (r'.*ful$', 'JJ'),
    (r'.*ious$', 'JJ'),
    (r'.*ble$', 'JJ'),
    (r'.*ic$', 'JJ'),
    (r'.*ive$', 'JJ'),
    (r'.*ic$', 'JJ'),
    (r'.*est$', 'JJ'),
    (r'^a$', 'PREP'),
]

brown_review_sents = nltk.corpus.brown.tagged_sents(categories=['reviews'])
brown_lore_sents = nltk.corpus.brown.tagged_sents(categories=['lore'])
brown_romance_sents = nltk.corpus.brown.tagged_sents(categories=['romance'])
 
brown_train = list(itertools.chain(brown_review_sents[:1000], brown_lore_sents[:1000], brown_romance_sents[:1000]))
brown_test = list(itertools.chain(brown_review_sents[1000:2000], brown_lore_sents[1000:2000], brown_romance_sents[1000:2000]))
 
conll_sents = nltk.corpus.conll2000.tagged_sents()
conll_train = list(conll_sents[:4000])
conll_test = list(conll_sents[4000:8000])
 
treebank_sents = nltk.corpus.treebank.tagged_sents()
treebank_train = list(treebank_sents)#[:1500])

train_sents = treebank_train


ubta_tagger = backoff_tagger(train_sents, [nltk.tag.UnigramTagger, nltk.tag.BigramTagger, nltk.tag.TrigramTagger, nltk.tag.AffixTagger])
ubat_tagger = backoff_tagger(train_sents, [nltk.tag.UnigramTagger, nltk.tag.BigramTagger, nltk.tag.AffixTagger, nltk.tag.TrigramTagger])
uabt_tagger = backoff_tagger(train_sents, [nltk.tag.UnigramTagger, nltk.tag.AffixTagger, nltk.tag.BigramTagger, nltk.tag.TrigramTagger])
aubt_tagger = backoff_tagger(train_sents, [nltk.tag.AffixTagger, nltk.tag.UnigramTagger, nltk.tag.BigramTagger, nltk.tag.TrigramTagger])


aubtr_tagger = nltk.tag.RegexpTagger(word_patterns, backoff=aubt_tagger)
raubt_tagger = backoff_tagger(train_sents, [nltk.tag.AffixTagger, nltk.tag.UnigramTagger, nltk.tag.BigramTagger, nltk.tag.TrigramTagger],
    backoff=nltk.tag.RegexpTagger(word_patterns))
 
ubt_tagger = backoff_tagger(train_sents, [nltk.tag.UnigramTagger, nltk.tag.BigramTagger, nltk.tag.TrigramTagger])
utb_tagger = backoff_tagger(train_sents, [nltk.tag.UnigramTagger, nltk.tag.TrigramTagger, nltk.tag.BigramTagger])
but_tagger = backoff_tagger(train_sents, [nltk.tag.BigramTagger, nltk.tag.UnigramTagger, nltk.tag.TrigramTagger])
btu_tagger = backoff_tagger(train_sents, [nltk.tag.BigramTagger, nltk.tag.TrigramTagger, nltk.tag.UnigramTagger])
tub_tagger = backoff_tagger(train_sents, [nltk.tag.TrigramTagger, nltk.tag.UnigramTagger, nltk.tag.BigramTagger])
tbu_tagger = backoff_tagger(train_sents, [nltk.tag.TrigramTagger, nltk.tag.BigramTagger, nltk.tag.UnigramTagger])



raubt_tagger = backoff_tagger(train_sents, [nltk.tag.AffixTagger,
    nltk.tag.UnigramTagger, nltk.tag.BigramTagger, nltk.tag.TrigramTagger],
    backoff=nltk.tag.RegexpTagger(word_patterns))

with open('raubt_pos_tagger.pickle','w') as f:
    f.write(pickle.dumps(raubt_tagger))

exit()

templates = [
    brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1,1)),
    brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (2,2)),
    brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1,2)),
    brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1,3)),
    brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (1,1)),
    brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (2,2)),
    brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (1,2)),
    brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (1,3)),
    brill.ProximateTokensTemplate(brill.ProximateTagsRule, (-1, -1), (1,1)),
    brill.ProximateTokensTemplate(brill.ProximateWordsRule, (-1, -1), (1,1))
]
 
trainer = brill.FastBrillTaggerTrainer(raubt_tagger, templates)
braubt_tagger = trainer.train(train_sents, max_rules=100, min_score=3)
