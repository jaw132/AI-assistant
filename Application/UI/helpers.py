from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

def run_spotify(artist, user, password):
    # Initiate the browser
    browser = webdriver.Chrome(ChromeDriverManager().install())

    browser.get('https://open.spotify.com/')

    # click on sign in button
    browser.find_element_by_class_name(
        '_3f37264be67c8f40fa9f76449afdb4bd-scss._1f2f8feb807c94d2a0a7737b433e19a8-scss').click()

    # Fill credentials
    browser.implicitly_wait(10)
    browser.find_element_by_name("username").send_keys(user)
    browser.find_element_by_name("password").send_keys(password)
    # Click Log In
    browser.find_element_by_id('login-button').click()

    browser.find_element_by_class_name('icon.search-icon').click()
    # browser.find_element_by_class_name('link-subtle._47872eede19af06725157ba21fea3516-scss').click()

    browser.find_element_by_class_name('_748c0c69da51ad6d4fc04c047806cd4d-scss').send_keys(artist)

    browser.find_element_by_class_name('_85fec37a645444db871abd5d31db7315-scss').click()

    browser.find_element_by_xpath(
        '//*[@id="main"]/div/div[2]/div[4]/main/div[2]/div[2]/div/div/div[2]/section/div/div[2]/div[2]/div/button[1]').click()

