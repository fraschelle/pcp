#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv

def load_csv(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        data_list = list(reader)

    results = []
    for i in range(1, len(data_list)):
        result = []
        for j in data_list[i]:
            if "." in j or "e" in j:
                try:
                    result.append(float(j))
                except ValueError:
                    result.append(j)
            else:
                try:
                    result.append(int(j))
                except ValueError:
                    result.append(j)

        results.append(result)

    return results, data_list[0]
