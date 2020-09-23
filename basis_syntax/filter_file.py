if __name__ == '__main__':
    with open('/Users/didi/weibao_std_marketing_crowd_20200901.txt') as f:
        with open("/Users/didi/weibao_std_marketing_crowd_20200902.txt", "w") as wf:
            wf.write('pid\n')

            found = False
            for line in f.readlines():
                if line.startswith('565948485011732'):
                    found = True
                if found:
                    wf.write(line)

    print('finished')
