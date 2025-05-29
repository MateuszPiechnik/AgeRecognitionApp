 # Analiza rozwiązań do szacowania wieku na podstawie wizerunku twarzy w oparciu o metody uczenia maszynowego

 ## 1. Przegląd litearury
 Na samym wstępie przeszukałem literaturę, która jest przydatna w tej pracy. Wiekszość prac znajduje się w folderze Literatura. Sa tam przykłady prac, zajmujące się tym samym problemem, wykorzystując różne modele i podejścia, a także artykuły z przeglądem najnowszych rozwiązań. 

 ## 2. Teoria
 Zapoznanie się z działaniem sieci neuronowych, a szczególniej sieciami CNN - dostosowane to pracy z obrazami. Wybranie kilku architektur sieci CNN (ResNet, EfficientNet), które zostaną użyte do przeprowadzenia eksperymentów, oraz zapoznanie sie z Vision Transformerami (ViT) - nowszym podejściem. 

 ## 3. Przygotwanie danych do treningu i implementacja
 Przegląd dostępnych zbiorów danych w sieci i ich analiza. Wybrałem zbiór UtkFace (https://susanqq.github.io/UTKFace/) i podzieliem w stosunku 80/10/10 na dane treningowe, walidacyjne i testowe. 
 
 Zaimplementowano ten sam preprocessing danych (rozmiar, augmentacja) dla wybranych modelów, aby mieć możliwość rzetelnego porównania. Kody do poszczególnych modeli w folderze ageEstimation. 

 Wybrane modele: 
 - ResNet50
 - EfficientNetB4
 - Vision Transformer (ViT)
 - Hybryda ResNet z ViT (na podstawie artykułu Age_Estimation_from_Face_Image_Leveraging_Concatenated_Features_of_Vision_Transformer_Along_with_Resnet-50.pdf)
 - Hybryda EfficientNet z ViT - zainspirowana hybryda wyzej
 - Hybryda EfficienNet z ViT ale ulepszona (są lepsze wyniki) - zmiana trochę architektury oraz niektórych hiperparametrów - przez pierwsze 5 epok trenuje tylko HEAD a dopiero potem trenowane są backbone'y tych dwóch architektur

Modele sa zapisywane na podstawie najniższego średniego błedu bezwzględnego uzyskanego na podstawie dancyh walidacyjnych. Zapisane są w folderze modele. 

## 4. Wyniki
Wyniki umieszczone są w folderze wyniki. Są tam wykresy dla pozsczególnych modeli a w pliku .csv dokładniejsze dane takie jake np. MAE i CS. 
 

## 5. Proponowany spis treści 

1.	Wstęp

    1.1.	Problem estymacji wieku	
    
    1.2.	Motywacja i zastosowania	

    1.3.	Cel i zakres pracy	
2.	Podstawy teoretyczne i przegląd literatury	

    2.1.	Podstawy uczenia maszynowego wykorzystywane do przetwarzania obrazów	

    2.2.	Ewolucja metod estymacji wieku	
    
    2.3.	Przegląd dostępnych rozwiązań	
3.	Metodyka badań	
    
    3.1.	Dostępne zbiory danych	
    
    3.2.	Preprocessing danych	
    
    3.3.	Metryki oceny	
    
    3.4.	Opis wybranych algorytmów testowych	
4.	Opis własnego algorytmu	
    
    4.1.	Podstawowa architektura i modyfikacje	
    
    4.2.	Szczegóły implementacyjne i wykorzystane frameworki	
    
    4.3.	Aplikacja	
5.	Eksperymenty i wyniki	
    
    5.1.	Parametry treningu, podział danych i dane testowe	
    
    5.2.	Wyniki algorytmów benchmarkowych na zbiorze testowym	
    
    5.3.	Wyniki autorskiego algorytmu na zbiorze testowym	
    
    5.4.	Analiza porównawcza wyników	
    
    5.5.	Interpretacja uzyskanych wyników	
6.	Podsumowanie	
    
    6.1.	Możliwości rozwoju i dalszych badań	
7.	Bibliografia	
