# https://www.r-bloggers.com/text-message-classification/

require(quanteda)
?quanteda 
require(RColorBrewer)
require(ggplot2)

fileName = "C:\\Users\\User\\PythonProjects\\ML-course-NN\\spam.csv"
spam=read.csv(fileName,header=TRUE, sep=",", quote='\"\"', stringsAsFactors=FALSE)

table(spam$v1)
names(spam)<-c("type","message")
#head(spam)

set.seed(2345)
spam<-spam[sample(nrow(spam)),] # перемешиваем набор случайным образом

msg.corpus<-corpus(spam$message)
docvars(msg.corpus) <- spam$type   #attaching the class labels to the corpus message text

# получаем корпус только из спама
spam.plot <- corpus_subset(msg.corpus, docvar1=="spam")

#now creating a document-feature matrix using dfm()
spam.plot <- dfm(spam.plot, tolower = TRUE, remove_punct = TRUE, 
                 remove_twitter = TRUE, remove_numbers = TRUE, remove=stopwords("SMART"))
spam_df <- as.data.frame(as.matrix(spam.plot))

spam.col <- brewer.pal(10, "BrBG")  

textplot_wordcloud(spam.plot, min.freq = 16, color = spam.col)  
title("Spam Wordcloud", col.main = "grey14")

# строим облако слов для нормальных сообщений
ham.plot<-corpus_subset(msg.corpus, docvar1=="ham")
ham.plot<-dfm(ham.plot,tolower = TRUE, remove_punct = TRUE, remove_twitter = TRUE, remove_numbers = TRUE,remove=c("gt", "lt", stopwords("SMART")))
ham.col=brewer.pal(10, "BrBG")  
textplot_wordcloud(ham.plot,min.freq=50,colors=ham.col,fixed.asp=TRUE)
title("Ham Wordcloud",col.main = "grey14")

#separating Train and test data
TRAIN_COUNT <- 4458
spam.train<-spam[1:TRAIN_COUNT,]
spam.test<-spam[TRAIN_COUNT:nrow(spam),]

msg.dfm <- dfm(msg.corpus, tolower = TRUE)  #generating document freq matrix
# отсекаем редко встречающиеся слова
msg.dfm <- dfm_trim(msg.dfm, min_termfreq = 5, min_docfreq = 3)  
# используем в качестве весов меру TF-IDF
msg.dfm <- dfm_tfidf(msg.dfm)

# преобразуем DFM в DataFrame
msg.dfm_df <- as.data.frame(as.matrix(msg.dfm))



# разделяем DFM на обучающую и тестовую выборки
msg.dfm.train<-msg.dfm[1:TRAIN_COUNT,]
msg.dfm.test<-msg.dfm[TRAIN_COUNT:nrow(spam),]

# создаём наивный байесовский классификатор
# передаём DFM и соотв. список меток
nb.classifier <- textmodel_nb(msg.dfm.train, spam.train[,1])
nb.classifier

# выполняем прогнозирование на тестовой выборке
pred <- predict(nb.classifier, msg.dfm.test)
pred_df <- as.data.frame(pred)

# строим матрицу
table(predicted=pred_df$pred, actual=spam.test[,1])

# точность классификатора на тестовых данных
mean(pred_df$pred == spam.test[,1])*100
