from typing import Generator, List

from mrjob.job import MRJob
from sklearn.cluster import Birch

FILE_NAME = "input.txt"
N_CLUSTERS = 5


class MRBIRCHClustering(MRJob):

    def mapper(self, _, line):
        # Розділення рядка на ключ і значення
        key, value = line.strip().split("\t")

        # Перетворення рядка на потрібні дані для кластеризації
        data_point = [float(coord) for coord in value.split(",")]

        # Вибір кластера для даної точки
        cluster_id = self.birch.predict([data_point])[0]

        # Вивід кластера та даної точки як результату
        yield cluster_id, data_point

    @staticmethod
    def reducer(cluster_id, data_points):
        cluster_points = [point for point in data_points]
        yield cluster_id, cluster_points

    def steps(self):
        return [self.mr(mapper=self.mapper, reducer=self.reducer)]


def read_file_data(file_name: str) -> List[Generator[float]]:
    data = []
    with open(file_name, "r") as file:
        for line in file:
            key, value = line.strip().split("\t")
            data.append((float(coord) for coord in value.split(",")))
    return data


def main():
    data = read_file_data(FILE_NAME)

    # Ініціалізація та навчання моделі BIRCH
    birch = Birch(n_clusters=N_CLUSTERS)
    birch.fit(data)

    # Запуск MapReduce задачі
    job = MRBIRCHClustering(args=["input.txt", "--jobconf", "mapreduce.job.reduces=2"])
    with job.make_runner() as runner:
        runner.run()
        for line in runner.stream_output():
            cluster_id, cluster_points = job.parse_output_line(line)
            print(f"Cluster {cluster_id}: {cluster_points}")


if __name__ == "__main__":
    main()
