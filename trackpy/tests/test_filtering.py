from pandas import DataFrame

from trackpy import filter_stubs, filter_clusters

import pytest

class TestFilterStubs:
    def test_cannot_filter_stubs_with_only_particle_col(self):
        tracks = DataFrame({'particle':[]})

        with pytest.raises(ValueError) as e:
            filter_stubs(tracks)

    def test_cannot_filter_stubs_with_only_frame_col(self):
        tracks = DataFrame({'frame':[]})

        with pytest.raises(ValueError) as e:
            filter_stubs(tracks)


    @pytest.mark.parametrize("frame_threshold,expected", [
        (1, [0, 1, 2, 3, 4, 5, 6]),
        (2, [0, 1, 2, 3, 4, 5, 6]),
        (3, [1, 2, 3, 4, 5, 6]),
        (5, [1, 2, 4, 5, 6]),
        (10, [4, 5, 6]),
        (50, [5, 6])])
    def test_removes_tracks_shorter_than_threshold(self, frame_threshold:int, expected:list[int]):
        track_lengths = [2, 9, 5, 4, 49, 50, 51]
        test_data = []
        for particle in range(7): 
            for frame in range(track_lengths[particle]):
                test_data.append({
                    'frame':frame,
                    'particle':particle
                })
        tracks = DataFrame.from_records(test_data)

        filtered_tracks: DataFrame = filter_stubs(tracks, frame_threshold)

        remaining_particles:list[int] = [int(particle) 
                                         for particle in filtered_tracks['particle'].unique()]
        assert remaining_particles == expected

class TestFilterClusters:
    def test_cannot_filter_clusters_if_frame_col_is_missing(self):
        tracks = DataFrame({'size': [], 'particle':[]})

        with pytest.raises(ValueError) as e:
            filter_clusters(tracks)

    def test_cannot_filter_clusters_if_size_col_is_missing(self):
        tracks = DataFrame({'particle': [], 'frame':[]})

        with pytest.raises(ValueError) as e:
            filter_clusters(tracks)

    def test_cannot_filter_clusters_if_particle_col_is_missing(self):
        tracks = DataFrame({'size': [], 'frame':[]})

        with pytest.raises(ValueError) as e:
            filter_clusters(tracks)

    @pytest.mark.parametrize("sizes,expected", [
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 1, 2, 3, 4, 5]),
        ([5, 2, 4, 1, 7, 10, 8, 3, 4, 9], [0, 1, 2, 3, 7, 8]),
        ([1, 1, 1, 4, 6, 10, 11, 15, 100, 1000], [0, 1, 2, 3, 4, 5]),
        ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1000], [])

    ])
    def test_removes_single_frame_particles_with_size_in_66th_percentile(self, sizes:list[int], expected:list[int]):
        tracks = DataFrame.from_records([{'particle':particle, 'size':size, 'frame':0}
                                         for particle, size in zip(range(10), sizes)])

        filtered_tracks:DataFrame = filter_clusters(tracks, quantile=0.66)

        remaining_particles:list[int] = [int(particle)
                                         for particle in filtered_tracks['particle'].unique()]
        assert remaining_particles == expected

    def test_removes_single_frame_particles_with_size_above_threshold(self):
        tracks = DataFrame.from_records([{'particle':particle, 'size':size, 'frame':0}
                for particle, size in zip(range(10), [1, 1, 100, 1, 1000, 20, 1, 1, 5, 1])])

        filtered_tracks:DataFrame = filter_clusters(tracks, threshold=10)

        remaining_particles:list[int] = [int(particle)
                                         for particle in filtered_tracks['particle'].unique()]
        assert remaining_particles == [0, 1, 3, 6, 7, 8, 9]

    def test_removes_particle_with_mean_size_above_threshold(self):
        small_particle_sizes: list[int] = [10-i for i in range(9)]
        large_particle_sizes: list[int] = [7+i for i in range(9)]
        sizes = small_particle_sizes + large_particle_sizes
        tracks = DataFrame({'particle': [0]*9 + [1]*9,
                            'frame': list(range(9))+list(range(9)),
                            'size': sizes})

        filtered_tracks = filter_clusters(tracks, threshold=10)

        remaining_particles:list[int] = [int(particle)
                                         for particle in filtered_tracks['particle'].unique()]
        assert remaining_particles == [0]