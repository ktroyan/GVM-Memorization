import pandas as pd

artists_df = pd.read_csv("./Scraping/googleart/Data/googleart_artists.csv", sep="\t")
print(artists_df.head())
artist_names = artists_df["artist_name"].values.tolist()

artist_names_filtered = ['Vincent van Gogh', 'Claude Monet', 'Rembrandt', 'Raphael', 'Paul Cézanne', 'Albrecht Dürer', 'Paul Gauguin', 'Gustav Klimt', 'Francisco Goya', 'Pierre-Auguste Renoir', 'Peter Paul Rubens', 'Frida Kahlo', 'Titian', 'Edgar Degas', 'Édouard Manet', 'Edvard Munch', 'Diego Rivera', 'Eugène Delacroix', 'Henri de Toulouse-Lautrec', 'William Blake', 'Paul Klee', 'El Greco', 'Sandro Botticelli', 'J. M. W. Turner', 'Gustave Courbet', 'Camille Pissarro', 'Giorgio Vasari', 'Michelangelo', 'Leonardo da Vinci', 'Egon Schiele', "Georgia O'Keeffe", 'Hokusai', 'Joaquín Sorolla', 'Jean-Baptiste-Camille Corot', 'Lucas Cranach the Elder', 'Mary Cassatt', 'François Boucher', 'Hans Holbein the Younger', 'Caspar David Friedrich', 'Zhao Mengfu', 'John Constable', 'Nicholas Roerich', 'Dante Gabriel Rossetti', 'Wassily Kandinsky', 'Giovanni Battista Tiepolo', 'Ernst Ludwig Kirchner', 'Honoré Daumier', 'Alfred Sisley', 'Johannes Vermeer', 'Odilon Redon', 'William Hogarth', 'James Abbott McNeill Whistler', 'Canaletto', 'Georges Seurat', 'Guido Reni', 'Paul Signac', 'Pieter Bruegel the Elder', 'Bartolomé Esteban Murillo', 'Jean-Honoré Fragonard', 'Jean-François Millet', 'Francisco de Zurbarán', 'Thomas Gainsborough', 'René Lalique', 'Giovanni Bellini', 'Jacques-Louis David', 'Berthe Morisot', 'Bertel Thorvaldsen', 'Sol LeWitt', 'Victor Vasarely', 'Jean Auguste Dominique Ingres', 'Ogata Kōrin', 'Eugène Boudin', 'Henry Fuseli', 'Jean-Antoine Watteau', 'Henri Rousseau', 'Albert Bierstadt', 'Gordon Parks', 'Luca Giordano', 'Edward Lear', 'Bronzino', 'Lucio Fontana', 'Gian Lorenzo Bernini', 'L. S. Lowry', 'John Everett Millais', 'Edward Hopper', 'Giovanni Bellini', 'Lovis Corinth', 'Franz Marc', 'Jusepe de Ribera', 'Hubert Robert', 'Hieronymus Bosch', 'Jean-Baptiste-Siméon Chardin', 'Julia Margaret Cameron', 'Harunobu Suzuki', 'Hubert Robert', 'Santiago Rusiñol', 'Jacob van Ruisdael', 'Gustave Caillebotte', 'Gerhard Richter', 'Ogata Kenzan', 'Ilya Repin', 'Gilbert Stuart', 'Lorenzo Lotto', 'Emperor Huizong of Song', 'Sidney Nolan', 'Wenceslaus Hollar', 'Michael Ancher', 'Giovanni Domenico Tiepolo', 'George Romney', 'Marià Fortuny', 'Qian Xuan', 'William Morris', 'Giotto', 'Anna Ancher', 'Marcantonio Raimondi', 'Johan Christian Dahl', 'Rogier van der Weyden', 'Giovanni Battista Piranesi', 'Stanford White', 'August Macke', 'Torii Kiyonaga', 'Toyohara Chikanobu', 'William-Adolphe Bouguereau', 'Walker Evans', 'Charles Willson Peale', 'Jan Brueghel the Elder', 'Arthur Streeton', 'Toyohara Kunichika', 'Keisai Eisen', 'Félicien Rops']

# convert the filtered list of artists (obtained through GPT-4 prompting) to a pandas dataframe
artist_names_filtered_df = pd.DataFrame(artist_names_filtered, columns=["artist_name"])

# create a new column in the artist_names_filtered_df dataframe with the number of paintings of each artist taken from the artists_df dataframe
artist_names_filtered_df["nb_artworks"] = artist_names_filtered_df["artist_name"].apply(lambda x: artists_df[artists_df["artist_name"] == x]["nb_artworks"].values[0])

# create a new column in the artist_names_filtered_df dataframe with the artist googleart url of each artist taken from the artists_df dataframe
artist_names_filtered_df["artist_url"] = artist_names_filtered_df["artist_name"].apply(lambda x: artists_df[artists_df["artist_name"] == x]["artist_url"].values[0])

# save the list of artists in a csv file
artist_names_filtered_df.to_csv("./Scraping/googleart/Data/googleart_artists_filtered_for_paintings.csv", sep="\t", index=False)