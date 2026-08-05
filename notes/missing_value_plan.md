## grammar_tags

Missing: 10.46%. Mechanism: MAR, tied to `pos`. Word classes that don't
inflect (conjunctions, interjections, some prepositions/adverbs) are ~100%
missing, inflecting classes (nouns, verbs, pronouns) are ~0%. Confirmed by
groupby(pos) missingness breakdown.

Decision: fill with a sentinel category "no_gram" rather than imputing a
plausible tag for these word classes, "no tag" is the correct value,
not an unknown one.