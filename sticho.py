import pandas as pd
import random
import re
from collections import defaultdict
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

import pprint
pp = pprint.PrettyPrinter(indent=4, depth=3)

class Sampler:
    '''
    Sampling from the corpus
    '''
    
    def __init__(self, data, authors=None, subcorpus=None, meters=None, n=100, min_rhymes=40, 
                 year_min=None, year_max=None, entire_work=False, single_poem=False, split_poems=False):
        '''
        Extract samples of specified length
        :: data        = input data
        :: authors     = list of authors to which the sampling is applied
        :: authors     = id of subcorpus to which the sampling is applied
        :: meters      = list of meters to which the sampling is applied (None if condition is not applied)
        :: n           = sample size measured by number of lines
        :: year_min    = lowest accepted year of publication
        :: year_max    = highest accepted year of publication
        :: min_rhymes  = minimum of rhymes required for sample (None if condition is not applied)
        '''
        
        # Container for samples
        self.samples_ = defaultdict(lambda: defaultdict(list))
        # Dict to store indices of current authors' samples
        sample_indices = defaultdict(int)
        
        # Iterate over poems
        for poem in data:
            a = poem['author']

            # Skip poem if year of publication does not match the conditions
            if year_min:
                if not poem['year']:
                    continue
                elif poem['year'] < year_min:
                    continue
            if year_max: 
                if not poem['year']:
                    continue
                if poem['year'] > year_max:
                    continue

            # Skip authors not included in the list
            if authors and a not in authors:
                continue
            
            # Skip poem not belonging to subcorpus
            if subcorpus and poem['subcorpus'] != subcorpus:
                continue
            
            # Use name of the poem as sample index if each sample should consist
            # of one poem only
            if single_poem == True:
                sample_indices[a] = poem['title']
                    
            # Iterate over lines
            for i, line in enumerate(poem['body']):
    
                # Skip lines that are not written in required metre
                if type(meters) is list and line['metre'] not in meters:
                    continue
    
                # Append line to current author's sample
                self.samples_[a][sample_indices[a]].append(line)
                
                # TODO
                # Rhyme has to be reindexed within a sample
                # This is just a workaround, which stores a copy of rhyming
                # lines directly into the line itself
                self.samples_[a][sample_indices[a]][-1]['rhyme_copy'] = list()
                for r in self.samples_[a][sample_indices[a]][-1]['rhyme']:
                    # Skip repetitions of the same word
                    if line['words'][-1]['token'].lower() == poem['body'][r]['words'][-1]['token'].lower():
                        continue
                    # Keep only rhymes pointing forward
                    if r < i:                        
                        continue
                    self.samples_[a][sample_indices[a]][-1]['rhyme_copy'].append({
                        'metre': poem['body'][r]['metre'],
                        'stress_pattern': poem['body'][r]['stress_pattern'],
                        'fin_word': poem['body'][r]['words'][-1],
                    })
                
                # If required sample size already reached, increase the index and skip to another poem
                if not entire_work and not single_poem and len(self.samples_[a][sample_indices[a]]) == n:
                    sample_indices[a] += 1
                    if not split_poems:
                        break
    
        # Drop samples that didn't reach the required size
        if not entire_work and not single_poem:
            for a in self.samples_:
                self.samples_[a] = {x:self.samples_[a][x] for x in self.samples_[a] if len(self.samples_[a][x]) >= n}        
                
        # Drop samples having less rhymes than required
        if min_rhymes:
            for a in self.samples_:
                rhymes_count = defaultdict(int)
                for s in self.samples_[a]:
                    for i,l in enumerate(self.samples_[a][s]):
                        rhymes_count[s] += len(l['rhyme_copy'])
                self.samples_[a] = {x:self.samples_[a][x] for x in self.samples_[a] if rhymes_count[x] >= min_rhymes}

    
    def level_samples(self, max_=None, exceptions=[]):
        '''
        Randomly drop samples in order to have equal amount of samples from each author.
        :: max_       = maximum number of samples per one author (None if condition is not applied)
        :: exceptions = list of authors to exclude from levelling
        '''
        
        # Find the author with lowest number of samples and compare it to max_ argument
        level_to = min([len(self.samples_[x]) for x in self.samples_ if x not in exceptions])
        if max_ and max_ < level_to:
            level_to = max_
        
        # Drop samples by random
        for a in self.samples_:
            while len(self.samples_[a]) > level_to:
                self.samples_[a].pop(random.choice(list(self.samples_[a].keys())))    

    def level_authors(self, n):
        '''
        Randomly drop authors in order to have n authors in the dataset
        :: n  = required number of authors
        '''
        
        # Keep all the samples if n >= number of authors
        if n >= len(self.samples_):
            return
    
        # Drop authors by random
        while len(self.samples_) > n:
            self.samples_.pop(random.choice(list(self.samples_.keys())))


class Features:
    '''
    Extract features from the samples
    '''
    
    def __init__(self, sampler, zscores=True):
        '''
        Get samples
        :: sampler  = instance of Sampler class or samples dict directly
        :: zscores  = transform to z-scores? (boolean)        
        '''

        if isinstance(sampler, dict):
            self.samples_ = sampler
        else:
            self.samples_ = sampler.samples_            
        self.df_ = pd.DataFrame()
        self.syll_peaks = "iye2E9{a&IYU1}@836Mu7oVOAQ0=ÓÉÁ"
        self.zscores_ = zscores
        
    def _to_dataframe(self, mft):
        '''
        Process dict of absolute frequencies and concat it with main dataframe
        :: zscores  = transform to z-scores? (boolean)
        :: mft      = most frequent types level
        '''
        
        # Transform dict into dataframe
        df = pd.DataFrame.from_dict(self.f_, orient='index').fillna(0)
              
        # Pick only most frequent types
        if mft > 0:
            most_frequent = df.sum().sort_values(ascending=False)[0:mft].index
            df = df[most_frequent]

        # Get the sample sizes    
        n = df.sum(axis=1)        

        # Relative frequencies
        df = df.div(n, axis=0)

        # Append to main dataframe
        self.df_ = pd.concat([self.df_, df], axis=1)
        
    def build_dataframe(self):
        '''
        Get relative frequencies and transform to z-scores if required
        '''

        # Replace NaN if present
        self.df_ = self.df_.fillna(0)

        # Transformation to z-scores (if required)
        if self.zscores_:
            self.df_ = (self.df_ - self.df_.mean())/self.df_.std(ddof=0)
            self.df_ = self.df_.fillna(0)


        
    def bow(self, domain='lemma', mft=100):
        '''
        Bag of words vectors
        :: domain   = lemma | word
        :: mft      = most frequent types level
        '''
    
        self.f_ = defaultdict(lambda: defaultdict(int))
        
        # Iterate over words
        for author in self.samples_:
            for sample in self.samples_[author]:
                for line in self.samples_[author][sample]:
                    for word in line['words']:
                        
                        # Count words/lemmata frequencies
                        if domain == 'word':                    
                            self.f_[(author, sample)]['wrd'+'_'+word['token']] += 1
                        elif domain == 'lemma':                    
                            self.f_[(author, sample)]['lem'+'_'+word['lemma']] += 1
                        else:
                            raise Exception('Invalid value for domain = {0}'.format(domain))
                
        # Build a dataframe
        self._to_dataframe(mft)
        
    def char_ngrams(self, n=3, blankspace=True, mft=100):
        '''
        Character n-grams vectors
        :: n           = ngram length
        :: blankspace  = include blankspaces? (boolean)
        :: mft         = most frequent types level        
        '''
    
        self.f_ = defaultdict(lambda: defaultdict(int))

        # Iterate over lines
        for author in self.samples_:
            for sample in self.samples_[author]:
                for line in self.samples_[author][sample]:

                    # Join words with underscore; append and prepend underscore
                    # to the resulting string
                    text_string = '_' + '_'.join([x['token'] for x in line['words']]) + '_'

                    # Delete blankspaces if required
                    text_string = re.sub('_*', '', text_string)

                    # Count ngrams frequencies
                    for i in range(0, len(text_string)-n+1):
                        self.f_[(author, sample)]['ngr'+str(n)+'_'+text_string[i:i+n]] += 1

        # Build a dataframe                                    
        self._to_dataframe(mft)
        
    def rhythmic_types(self, mft=100):
        '''
        Rhythmic types vectors
        :: mft      = most frequent types level
        '''
    
        self.f_ = defaultdict(lambda: defaultdict(int))

        # Iterate over lines
        for author in self.samples_:
            for sample in self.samples_[author]:
                for line in self.samples_[author][sample]:

                    # Count rhythmic types frequencies
                    self.f_[(author, sample)]['rht_'+line['metre']+'_'+line['stress_pattern']] += 1

        # Build a dataframe                                    
        self._to_dataframe(mft)
        
    def rhythmic_ngrams(self, n=3, mft=100):
        '''
        Rhythmic ngrams vectors
        :: n        = ngram length
        :: mft      = most frequent types level
        '''
    
        self.f_ = defaultdict(lambda: defaultdict(int))

        # Iterate over lines
        for author in self.samples_:
            for sample in self.samples_[author]:
                for line in self.samples_[author][sample]:
                    
                    # Count rhythmic ngrams frequencies
                    for i in range(0, len(line['stress_pattern'])-n+1):
                        self.f_[(author, sample)]['ngt'+str(n)+'_'+line['metre']+str(i)+'_'+line['stress_pattern'][i:i+n]] += 1                    

        # Build a dataframe                                    
        self._to_dataframe(mft)
        
    def _split_to_snd_clusters(self, xsampa):
        '''
        Split xsampa representation to nucleus and onset/coda
        :: xsampa  = word in xsampa
        '''
        
        xsampa = re.sub(r'(['+self.syll_peaks+'])', r'#\1#', xsampa)
        xsampa = re.sub('##', '#∅#', xsampa)
        xsampa = re.sub('#$', '#∅', xsampa)
        xsampa = re.sub('^#', '∅#', xsampa)
        xsampa = xsampa.split('#')
        return xsampa

    def _count_syllables(self, xsampa):
        '''
        Count syllables in xsampa representation of the word
        :: xsampa  = word in xsampa
        '''
        
        xsampa = re.sub('['+self.syll_peaks+']', r'#', xsampa)
        return xsampa.count('#')
        
        
    def rhyme_profile(self, method='word_length', mft=True,
                      snd_position=1, ending='m'):
        '''
        Rhyme vectors
        :: method         = word length | stress | pos | sounds
        :: mft            = most frequent types level
        :: snd_position   = which sound slot to analyze
                            1: coda of final syllable
                            2: nucleus of final syllable
                            3: praetura of final syllable + coda of penultimate syllable
                            4: nucleus of final syllable
        '''
        
        self.f_ = defaultdict(lambda: defaultdict(int))

        # Iterate over lines
        for author in self.samples_:
            for sample in self.samples_[author]:
                for line in self.samples_[author][sample]:

                    # Skip if not required line ending
                    if ending:
                        current_ending = line['metre'][-1]
                    else:
                        current_ending = '0'
                    if ending and line['metre'][-1] not in ending:
                        continue
                    
                    

                    # Iterate over rhymes
                    for rhyme in line['rhyme_copy']:
                        
                        # Word lengths                        
                        if method == 'word_length':
                            val1 = self._count_syllables(line['words'][-1]['xsampa'])
                            val2 = self._count_syllables(rhyme['fin_word']['xsampa'])
                            val = '-'.join(sorted([str(val1), str(val2)]))
                            self.f_[(author, sample)]['rhw'+current_ending+'_'+val] += 1
                        
                        # Stress position
                        if method == 'stress':
                            simple_pattern1 = re.sub('[iI]', '1', line['stress_pattern'])                            
                            simple_pattern1 = re.sub('[oO]', '0', simple_pattern1)                            
                            simple_pattern2 = re.sub('[iI]', '1', rhyme['stress_pattern'])                            
                            simple_pattern2 = re.sub('[oO]', '0', simple_pattern2)                            
                            val1 = re.sub(r'^.*(10*)$', r'\1', simple_pattern1)
                            val2 = re.sub(r'^.*(10*)$', r'\1', simple_pattern2)
                            val = '-'.join(sorted([str(len(val1)), str(len(val2))]))
                            self.f_[(author, sample)]['rhs'+current_ending+'_'+val] += 1

                        # POS pair
                        elif method == 'pos':
                            val1 = line['words'][-1]['tag']
                            val2 = rhyme['fin_word']['tag']
                            val = '-'.join(sorted([val1, val2]))
                            self.f_[(author, sample)]['rhp'+current_ending+'_'+val] += 1
                            
                        # Sound frequencies
                        elif method == 'sounds':
                            val1 = self._split_to_snd_clusters(line['words'][-1]['xsampa'])
                            val2 = self._split_to_snd_clusters(rhyme['fin_word']['xsampa'])
                            if len(val1) < snd_position or len(val2) < snd_position:
                                continue
                            val1 = val1[(-1)*snd_position]
                            val2 = val2[(-1)*snd_position]
                            val = '-'.join(sorted([val1, val2]))
                            self.f_[(author, sample)]['rhx'+str(snd_position)+current_ending+'_'+val] += 1
 
        # Build a dataframe                                    
        self._to_dataframe(mft)

    def sound_frequencies(self, mft=100):
        '''
        Character n-grams vectors
        :: mft         = most frequent types level        
        '''
    
        self.f_ = defaultdict(lambda: defaultdict(int))

        # Iterate over lines
        for author in self.samples_:
            for sample in self.samples_[author]:
                for line in self.samples_[author][sample]:

                    # Join xsampa of words with underscore
                    xsampa_string = ''.join([x['xsampa'] for x in line['words']])

                    #
                    #print(xsampa_string)
                    xsampa_string = ('#').join([x for x in xsampa_string])
                    xsampa_string = re.sub(r'#\\', r'\\', xsampa_string)
                    xsampa_string = re.sub(r'#:', r':', xsampa_string)
                    #print(xsampa_string);input()
                    sounds = xsampa_string.split('#')

                    # Count ngrams frequencies
                    for sound in sounds:
                        self.f_[(author, sample)]['snd'+'_'+sound] += 1

        # Build a dataframe                                    
        self._to_dataframe(mft)
        
class Classification:
    '''
    #TODO
    '''
    
    def __init__(self, f, clf, **kwargs):
        '''
        Initialize the classifier
        :: f         = instance of Features class or dataframe directly
        :: clf       = classifier to be used; options:
                       (1) 'svm' for SVM with one-vs-one strategy
                       (2) 'svm_ovr' for SVM with one-vs-rest strategy
                       (3) 'delta' for K nearest neighbor
        :: **kwargs  = arguments to selected classifier
        '''

        # Get the dataframe
        if isinstance(f, Features):
            self.df_ = f.df_
        else:
            self.df_ = f

        # Initialize selected classifier
        if clf == 'svm':
            self.clf = SVC(**kwargs)
        elif clf == 'svm_ovr':
            self.clf = OneVsRestClassifier(SVC(**kwargs))
        elif clf == 'delta':
            self.clf = KNeighborsClassifier(**kwargs)
        elif clf == 'delta':
            self.clf = OneClassSVM(**kwargs)

    def _reduce_training_set(self, train_a, train_v, targ_a):
        '''
        Reduce training set in order to have equal number of samples
        per author during cross validation
        '''

        samples_dict = defaultdict(list)
        training_vectors = list()
        training_authors = list()
        to_skip = list()

        for i, a in enumerate(train_a):
            if a != targ_a:
                samples_dict[a].append(i)
                
        for a in samples_dict:
            r = random.choice(samples_dict[a])
            to_skip.append(r)

        for i, a in enumerate(train_a):
            if i not in to_skip:
                training_authors.append(train_a[i])
                training_vectors.append(train_v[i])

        return training_authors, training_vectors            
            
    def cross_validation(self, level_samples=True):
        '''
        Cross-validate the dataframe
        '''

        # Create container for results
        self.results = {
            'accuracy': 0,
            'decisions': dict(),
        }

        # Loop through samples
        for row in self.df_.iterrows():
            index, data = row

            # Pick current sample as target and use the rest for training
            target_vector = [data.tolist()]
            target_author = index[0]
            training_vectors = self.df_.drop(index).values.tolist()
            training_authors = self.df_.drop(index).index.get_level_values(0).tolist()

            # Level the number of training samples
            if level_samples:
                training_authors, training_vectors = self._reduce_training_set(training_authors, 
                                                                               training_vectors, 
                                                                               target_author)
            # Train the model
            self.clf.fit(training_vectors, training_authors)
            # Classify the target
            predicted = self.clf.predict(target_vector)
            # Increase accuracy if classification was successful
            if target_author == predicted:
                self.results['accuracy'] += 1
            # Store the decision
            self.results['decisions'][index] = predicted[0]

        # Accuracy to relative numbers
        self.results['accuracy'] /= len(self.df_.index)
        
        return self.results['accuracy']

    def feature_importances(self):
        '''
        Linear support vector machine feature weights
        '''
        # TODO: This really needs some optimization!
        X = self.df_.values.tolist()
        Y = self.df_.index.get_level_values(0).tolist()
        feature_names = self.df_.columns.values
        author_names = list(self.df_.index.get_level_values(0).unique())
        self.clf.fit(X, Y)
        if len(author_names) == 2:
            fi = pd.DataFrame(self.clf.coef_, columns=feature_names, index=['w'])
        else:
            fi = pd.DataFrame(self.clf.coef_, columns=feature_names, index=author_names)
        return fi.sort_index(axis=0).sort_index(axis=1)            