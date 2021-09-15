#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include "csvstream.h"

using namespace std;


class Classifier {
public:
    // default ctor
    Classifier();

    Classifier(vector<pair<string, string>>& train_data_in,
        bool debug_in);

    size_t get_num_post() const;

    size_t get_word_size() const;

    // REQUIRES:
    // MODIFIES:
    // EFFECTS: train the classifier
    void train();
    // REQUIRES:
    // MODIFIES:
    // EFFECTS: make predictions for new piazza posts
    void predict(vector<pair<string, string>>& test_data);

    void print_log_prior();

    void print_log_likelihood();

    pair<string, double> prediction(const set<string>& test_content);

private:

    // EFFECTS: Returns a set containing the unique "words" in the original
    //          string, delimited by whitespace.
    set<string> unique_words(const string& str);
    // EFFECTS: For each word w, the number of posts in the entire
    // training set that contain w
    void num_posts_word_impl();
    // EFFECTS: For each label C, the number of posts with that label C
    void num_posts_label_impl();
    // EFFECTS: For each label C and word w, the number of posts with
    // label C that contain w
    void num_posts_label_word_impl();
    //
    void log_prior_impl();
    //
    void log_likelihood_impl();

    // training data set
    vector<pair<string, string>> train_data;
    // member variables
    bool debug;
    // all words in data set
    string words;
    // all labels in data set
    string labels;
    // The set of unique words in the entire training set.
    set<string> unique_word;
    // The set of unique label in the entire training set.
    set<string> unique_label;
    // The total number of posts in the entire training set
    size_t num_post;
    // The number of unique words in the entire training set.
    size_t word_size;
    // set of each unique word and label in each post
    multiset<pair<string, string>> label_word;
    // all labels in data set
    multiset<string> labels_set;
    // For each word w, the number of posts in the entire
    // training set that contain w
    map<string, double> num_posts_word;
    // For each label C, the number of posts with that label C
    map<string, double> num_posts_label;
    // For each label C and word w, the number of posts with
    // label C that contain w
    map<pair<string, string>, double> num_posts_label_word;

    map<string, double> log_prior;

    map<pair<string, string>, double> log_likelihood;
};

// Constructor
Classifier::Classifier(vector<pair<string, string>>& train_data_in,
    bool debug_in) :
    train_data(train_data_in), debug(debug_in) {
    num_post = train_data_in.size();
}

// EFFECTS: Returns a set containing the unique "words" in the original
//          string, delimited by whitespace.
set<string> Classifier::unique_words(const string& str) {
    istringstream source(str);
    set<string> words;
    string word;
    // Read word by word from the stringstream and insert into the set
    while (source >> word) {
        words.insert(word);
    }
    return words;
}

void Classifier::train() {
    if (debug) {
        cout << "training data:" << endl;
    }
    for (const auto& data : train_data) {
        if (debug) {
            cout << "  label = " << data.first
                << ", content = " << data.second << endl;
        }
        words = words + " " + data.second;
        labels = labels + " " + data.first;
        labels_set.insert(data.first);
        set<string> words_set = unique_words(data.second);
        for (const auto& word : words_set) {
            label_word.insert({ data.first, word });
        }
    }
    unique_word = unique_words(words);
    unique_label = unique_words(labels);
    word_size = unique_word.size();

    num_posts_word_impl();
    num_posts_label_impl();
    log_prior_impl();
    num_posts_label_word_impl();
    log_likelihood_impl();
}

// EFFECTS: For each word w, the number of posts in the entire
// training set that contain w
void Classifier::num_posts_word_impl() {
    pair<string, string> lw;
    for (const auto& word : unique_word) {
        double counts = 0.0;
        for (const auto& label : unique_label) {
            lw = make_pair(label, word);
            counts += static_cast<double>(label_word.count(lw));
        }
        num_posts_word.insert({ word, counts });
    }
}

// EFFECTS: For each label C, the number of posts with that label C
void Classifier::num_posts_label_impl() {
    for (const auto& label : unique_label) {
        double counts = static_cast<double>(labels_set.count(label));
        num_posts_label.insert({ label, counts });
    }
}

void Classifier::log_prior_impl() {
    for (const auto& label : unique_label) {
        double prob = log(num_posts_label[label] / num_post);
        log_prior.insert({ label, prob });
    }
}

// EFFECTS: For each label C and word w, the number of posts with
// label C that contain w
void Classifier::num_posts_label_word_impl() {
    pair<string, string> lw;
    for (const auto& label : unique_label) {
        for (const auto& word : unique_word) {
            lw = make_pair(label, word);
            double counts = static_cast<double>(label_word.count(lw));
            num_posts_label_word.insert({ lw, counts });
        }
    }
}

void Classifier::log_likelihood_impl() {
    for (const auto& lw : num_posts_label_word) {
        double log_prob;
        if (lw.second != 0.0) {
            log_prob =
                log(lw.second / num_posts_label[lw.first.first]);
            log_likelihood.insert({ lw.first, log_prob });
        }
        else {
            log_prob =
                log(num_posts_word[lw.first.second] / num_post);
            log_likelihood.insert({ lw.first, log_prob });
        }
    }
}

void Classifier::print_log_prior() {
    cout << endl;
    cout << "classes:" << endl;
    for (const auto& it : log_prior) {
        cout << "  " << it.first << ", " << num_posts_label[it.first]
            << " examples, log-prior = " << it.second << endl;
    }
}

void Classifier::print_log_likelihood() {
    cout << "classifier parameters:" << endl;
    for (const auto& it : log_likelihood) {
        if (num_posts_label_word[it.first]) {
            cout << "  " << it.first.first << ":"
                << it.first.second << ", count = "
                << num_posts_label_word[it.first]
                << ", log-likelihood = " << it.second << endl;
        }
    }
}

void Classifier::predict(vector<pair<string, string>>& test_data) {
    set<string> unique_word_test;
    int correct = 0;
    for (const auto& it : test_data) {
        unique_word_test = unique_words(it.second);
        pair<string, double> pred = prediction(unique_word_test);
        cout << "  correct = " << it.first << ", predicted = "
            << pred.first << ", log-probability score = "
            << pred.second << endl << "  content = "
            << it.second << endl << endl;
        if (it.first == pred.first) correct++;
    }
    cout << "performance: " << correct << " / "
        << static_cast<int>(test_data.size())
        << " posts predicted correctly" << endl;
}

pair<string, double> Classifier::prediction(const set<string>& test_content) {
    map<string, double> pred;
    for (const auto& label : unique_label) {
        double score = log_prior[label];
        for (const auto& word : test_content) {
            if (unique_word.find(word) == unique_word.end()) {
                score += log(1.0 / num_post);
            }
            else {
                score += log_likelihood[{label, word}];
            }
        }
        pred.insert({ label, score });
    }
    double max_score = pred.begin()->second;
    pair<string, double> result =
        make_pair(pred.begin()->first, pred.begin()->second);
    for (const auto& it : pred) {
        if (it.second > max_score) {
            max_score = it.second;
            result = make_pair(it.first, it.second);
        }
    }
    return result;
}

size_t Classifier::get_num_post() const {
    return num_post;
}

size_t Classifier::get_word_size() const {
    return word_size;
}

int argument_check(int argc, char* argv[], bool& debug) {
    if ((argc != 3) && (argc != 4)) {
        cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
        return -1;
    }
    if (argc == 4) {
        string debug_str = argv[3];
        if (debug_str == "--debug") {
            debug = true;
        }
        else {
            cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
            return -1;
        }
    }
    return 0;
}

int main(int argc, char* argv[]) {
    cout.precision(3);
    string train_filename = argv[1];
    string test_filename = argv[2];
    bool debug = false;
    vector<pair<string, string>> train_data;
    vector<pair<string, string>> test_data;

    // Error checking -- command line arguments
    if (argument_check(argc, argv, debug) == -1) return -1;

    // Error checking -- file open and read in
    try {
        // read in training data
        csvstream csv_train_in(train_filename);
        map<string, string> train_data_in;
        while (csv_train_in >> train_data_in) {
            string tag = train_data_in["tag"];
            string content = train_data_in["content"];
            train_data.push_back(make_pair(tag, content));
        }
        // read in test data
        csvstream csv_test_in(test_filename);
        map<string, string> test_data_in;
        while (csv_test_in >> test_data_in) {
            string tag = test_data_in["tag"];
            string content = test_data_in["content"];
            test_data.push_back(make_pair(tag, content));
        }
    }
    catch (const csvstream_exception& e) {
        cout << e.what() << endl;
    }

    Classifier classifier(train_data, debug);

    classifier.train();

    cout << "trained on " << classifier.get_num_post()
        << " examples" << endl;
    if (debug) {
        cout << "vocabulary size = " << classifier.get_word_size() << endl;
        classifier.print_log_prior();
        classifier.print_log_likelihood();
    }
    cout << endl << "test data:" << endl;
    classifier.predict(test_data);

    return 0;
}
