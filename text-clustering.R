install.packages("tm")
install.packages("RODBC")
install.packages("SnowballC")
install.packages("textcat")
install.packages("RTextTools")
install.packages("tidytext")
install.packages("wordcloud")
install.packages("BBmisc")

library(tm)
#library(RODBC)
library(SnowballC)
library(textcat)
library(RTextTools)
library(tidytext)
library(wordcloud)
library(BBmisc)

text <- readLines("C:\\Users\\User\\PythonProjects\\ML-course-NN\\rus-text.txt", 
                  n = -1, encoding = "UTF-8")
text[1]

# --- предобработка (очистка) текста
#text <- gsub("[^[:alnum:]]", " ", text) # удаление всех неалфавитных и нецифровых символов (только для англ. яз.)
text <- gsub("[[:punct:]]", " ", text) # удал-е знаков препинаний
text <- gsub("[0-9]+", "", text) # удал-е цифр
text <- tolower(text)
text <- removeNumbers(text)
text <- removeWords(text, stopwords("russian")) # english russian - удаление стоп-слов

vec_text <- as.vector(text)
vec_text

corpus <- Corpus(VectorSource(vec_text)) # формируем корпус текстов
corpus
dtm <- DocumentTermMatrix(corpus) # 
dtm
tdm <- TermDocumentMatrix(corpus) # формируем текст-документную матрицу
tdm

word_matrix <- as.matrix(tdm) # преобразуем TDM в простую матрицу
words_freq <- sort(rowSums(word_matrix), decreasing = TRUE) # сортируем по убыванию частоты
words_freq

words_freq <- data.frame(freq = words_freq, word = names(words_freq)) # преобразуем марицу в дата-фрейм

wordcloud(words = words_freq$word, freq = words_freq$freq, 
          scale=c(4,.4), min.freq = 2, max.words=Inf, random.order=FALSE, 
          rot.per=0.1, ordered.colors=FALSE, random.color=TRUE, colors=brewer.pal(8, "Dark2"))

# удаление пустых строк - термов, которые не встречаются ни разу в текстах корпуса
rowTotals <- apply(dtm, 1, sum)
dtm <- dtm[rowTotals > 0, ]
m <- as.matrix(dtm)
str(m)
rownames(m) < -1 : nrow(m)

# нормализация матрицы DTM
norm_eucl <- function(m)   # ф-ция для вычисления Евклидовой нормы
  m / apply(m, 1, function(x) sum(x ^ 2) ^ 0.5)

m_norm <- norm_eucl(m)

# кластеризация методом к-средних
clust <- kmeans(m_norm, 3, 20)

text[clust$cluster == 1]
