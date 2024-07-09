from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import pandas as pd
import time

# Khởi tạo Webdriver
chromedriver_path = r'../chromedriver-win64/chromedriver.exe'
service = Service(chromedriver_path)
driver = webdriver.Chrome(service=service)

url = 'https://www.thegioididong.com/dtdd'
driver.get(url)

data = []  #Khởi tạo biến lưu trữ các liên kết

try:
    # Load sản phẩm
    WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'item')))
    
    # Lấy các URL sản phẩm chính
    while len(data) < 90:
        # Lướt xuống cuối trang
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        # Thời gian chờ trang sản phẩm load
        time.sleep(5) 
        
        # Lấy dữ liệu link, loại bỏ link có đầu là topzone.vn
        products = driver.find_elements(By.CLASS_NAME, 'item')
        for product in products:
            try:
                a_tag = product.find_element(By.TAG_NAME, 'a')
                href = a_tag.get_attribute('href')
                if 'topzone.vn' not in href and href.startswith('https://www.thegioididong.com/dtdd'):  # Ensure it's a valid product link
                    data.append({'product_url': href, 'reviews': []})
            except Exception as e:
                print(f"Error getting product link: {e}")
                
            # Kiểm tra xem đã đủ dữ liệu (ở đây lấy 90 link)
            if len(data) >= 90:
                break
        
        # Tìm và bấm vào nút "Xem thêm" nếu nó tồn tại
        try:
            show_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, 'view-more'))
            )
            show_more_button.click()
        except Exception as e:
            print(f"No more 'Xem thêm' button found or an error occurred: {e}")
            break

    # Lấy đánh giá và số sao đã đánh giá, người dùng đã đánh giá
    for product_data in data.copy():
        review_url = product_data['product_url'] + '/danh-gia'
        driver.get(review_url)  # Truy cập trang đánh giá
        
        # Kiểm tra trang có /danh-gia hay không
        if '/danh-gia' in driver.current_url:
            # Lấy đánh giá tối đa là 50 trang
            max_pages = 50
            page_num = 1
            
            while page_num <= max_pages:
                if page_num == 1:
                    review_page_url = review_url
                else:
                    review_page_url = f"{review_url}?page={page_num}"
                
                driver.get(review_page_url)
                
                # Kiểm tra xem đã hết trang /danh-gia hay không
                if '/danh-gia' not in driver.current_url:
                    break
                
                # Kiểm tra trang có bị lỗi 404 không
                if '404' in driver.title or 'Page not found' in driver.page_source:
                    break
                
                # Lấy đánh giá và số sao đã đánh giá
                reviews = driver.find_elements(By.CLASS_NAME, 'par')
                for review in reviews:
                    try:
                        star_rating_elements = review.find_elements(By.CLASS_NAME, 'iconcmt-starbuy')
                        star_rating = len(star_rating_elements)
                        comment = review.find_element(By.CLASS_NAME, 'cmt-txt').text.strip()
                        product_data['reviews'].append({'comment': comment, 'star_rating': star_rating})
                    except Exception as e:
                        print(f"Error getting review: {e}")
                
                # Kiểm tra và dừng nếu không có thêm đánh giá nữa
                if len(reviews) == 0:
                    break
                
                # Tăng số trang lên 1
                page_num += 1
        
        # Nếu không có /danh-gia hoặc không có trang đánh giá nào, bỏ qua
        else:
            continue

except Exception as e:
    print(f"Error: {e}")

finally:
    driver.quit()

# Lưu dữ liệu đánh giá vào DataFrame
review_data = []
for item in data:
    for review in item['reviews']:
        review_data.append({
            'Product URL': item['product_url'],
            'Comment': review['comment'],
            'Star Rating': review['star_rating']           
        })

df = pd.DataFrame(review_data)
df.to_csv('product_reviews2.csv', index=False, encoding='utf-8')

print("Đã lưu các đánh giá và số sao đã đánh giá vào file product_reviews.csv")
