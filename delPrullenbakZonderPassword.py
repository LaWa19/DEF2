# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:06:03 2019
"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
import selenium.webdriver.support.ui as ui
import selenium.webdriver.support.expected_conditions as EC
#import os
import time
import pyautogui
#import simplejson
import glob
import datetime
import matplotlib.pyplot as plt
import urllib
import math
import numpy as np

wachtwoorden=[]
pyautogui.PAUSE = 0.1
##pyautogui.click(100, 100)
##pyautogui.moveTo(100, 150)
##pyautogui.moveRel(0, 10)  # move mouse 10 pixels down
##pyautogui.dragTo(100, 150)
##pyautogui.dragRel(0, 10)


class LaPrul():
    def __init__(self):
        options=webdriver.ChromeOptions()
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--ingore-ssl-erros")
        ##dir_path=os.path.dirname(os.path.realpath(__file__))
        ##chromedriver=dir_path+"/chromedriver"
        ##os.environ["webdriver.chrome.driver"]=chromedriver
        self.driver=webdriver.Chrome(chrome_options=options)
        #pyautogui.click(980, 25,button="left")
        #driver=webdriver.Chrome()
        self.goToStack()
        self.prul()
    def typen(self,woord):
        time.sleep(1)
        for letter in woord:
            pyautogui.keyDown(str(letter))
            pyautogui.keyUp(str(letter))
    def pressenter(self):
        pyautogui.keyDown("enter")
        pyautogui.keyUp("enter")
    def timerPractice(self):
        time.sleep(5)
            
    def goToStack(self):
        website="https://roelandnaaktgeboren.stackstorage.com/login"
        self.driver.get(website)
        
        username=input("geef username van netwerkschijf")
        password=input("geef password van netwerkschijf")
        #inloggen
        ui.WebDriverWait(self.driver,15).until(EC.presence_of_element_located((By.XPATH, "/html/body/div/div/div/form/div[1]/input[1]"))).send_keys(username)
#        self.typen("def")
        ui.WebDriverWait(self.driver,15).until(EC.presence_of_element_located((By.XPATH, "/html/body/div/div/div/form/div[1]/input[2]"))).send_keys(password)
#        self.typen("speedcam")
        ui.WebDriverWait(self.driver,15).until(EC.presence_of_element_located((By.XPATH, "/html/body/div/div/div/form/div[1]/button"))).click()
#        expediaUnitTest.pressenter(self)
        ui.WebDriverWait(self.driver,15).until(EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div[1]/nav[1]/div/ul/li[5]/a"))).click()
        
#        website="https://roelandnaaktgeboren.stackstorage.com/trashbin"
#        self.driver.get(website)
#        time.sleep(15)
    def prul(self):
        ui.WebDriverWait(self.driver,15).until(EC.presence_of_element_located((By.XPATH, "/html/body/div/div[3]/div[2]/div[1]/div/a"))).click()
        ui.WebDriverWait(self.driver,15).until(EC.presence_of_element_located((By.XPATH, "/html/body/div[2]/div/div/div/div[3]/button[2]"))).click()

        
    def teardown(self):
        self.driver.close()
        self.driver.quit()




    

