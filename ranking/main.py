import train


if __name__ == '__main__':
    train.set_seed(0xDEAD)
    while True:
        mode = input("Выберите дальнейшее действие:\ntrain - выполнить обучение модели;\nget_ndcg - вывести значение "+
                     "NDCG для тестового набора данных;\nexit - выйти из программы\n\nВвод: ")
        if mode == 'train':
            file_name = input("Укажите полный путь до файла с обучающими данными: ")
            print(train.train_model(file_name))
        if mode == 'get_ndcg':
            file_name = input("Укажите полный путь до файла с тестовыми данными: ")
            print(train.get_ndcg(file_name))
        if mode == 'exit':
            break
        print()
