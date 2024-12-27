'''Webscrapper to keep track of our classmates scores'''
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options

def check_all_found():
    for user in users:
        if 'score' not in users[user]:
            return False
    return True

options = Options()
options.add_argument("--headless")  
options.add_argument("--disable-gpu") 
options.add_argument("--window-size=1920,1080")  

driver = webdriver.Edge(options=options)


driver.implicitly_wait(10)

users = {"gatopizza":{}, "covid-24":{}, "happy beavers":{}, "cross validation uclm":{}, "computofilos":{}, "clausulas del exito":{}, "ml and prolog enjoyers":{}}

print("Searching...", end="")
for i in range(1, 27):
    # print(f"==========Pagina {i}==========")
    print(".", end="")
    if check_all_found():
        break
    driver.get(f"https://www.drivendata.org/competitions/66/flu-shot-learning/leaderboard/?page={i}")

    spans = driver.find_elements(By.XPATH, "//table//td[3]//span[@data-bs-original-title='This is a team']")

    for span in spans:
        team_name = span.text.lower()
        #print(team_name)
        if team_name in users:
            row = span.find_element(By.XPATH, "./ancestor::tr")
            puntuacion = row.find_element(By.XPATH, ".//td[4]").text
            rank = row.find_element(By.XPATH, ".//td[1]").text
            users[team_name]['score'] = puntuacion
            users[team_name]['rank'] = rank

with open("results.txt", "w") as f:
    for user in users:
        f.write(f"-------Team: {user}--------\n")
        if 'score' in users[user]:
            f.write(f"Rank: {users[user]['rank']}\n")
            f.write(f"Score: {users[user]['score']}\n")
        f.write("\n")