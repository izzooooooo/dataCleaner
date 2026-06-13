toolun lokal olaraq terminalda acilmasi ucun lazım olanlar:
1.      terminalda cd "folder path"
	pip install -r requirements.txt
səbəb isə istifadə edilmiş kitabxanaların lokala yuklenmesi ve rahatliqla lokalda acila bilməsidir.
2. bütün kitabxanalar yuklendikdən sonra:
	streamlit run app.py
3. artiq lokalda proqram işləyir
4.(optional) əgər 200MB-dan cox upload limitine ehtiyac varsa streamlit run app.py --server.maxUploadSize 300(optional limit MB)
qeyd:200+ upload da ai suggestions ve cleansing funksiyalarinin xeta verme riski var. 
//hal hazirda test edilmiş ən böyük data:https://archive.ics.uci.edu/dataset/791/metropt+3+dataset 1.51 Mln rows 208MB