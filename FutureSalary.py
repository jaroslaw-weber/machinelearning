import random
import names
import numpy as np
from sklearn.linear_model import LinearRegression
from math import floor
#model calculating future salary using linear regression

# preparing data
# data size
data_len = 50


def randomkid():
    name = names.get_full_name()
    sex = random.randint(0, 1)  # 1 is a boy, 0 is a girl
    iq = random.randint(10, 120)  # kid's iq
    rich_family = random.randint(0, 1)  # 1 is rich, 0 is poor
    healthy = random.randint(0, 100)  # % of healthy
    return [name, sex, iq, rich_family, healthy]


# get random data
KIDS = [randomkid() for i in range(data_len)]
print("random kids: {0}".format(KIDS[:3]))
# data preprocessing -> removing names
prepX = [k[1:] for k in KIDS]
X = np.array(prepX, dtype=np.float32)  # creating numpy array
print("random input: {0}".format(X[:3]))

#lets say there is a way to calculate future salary
#kids born in rich families and having high iq tends to get better salary.
#man gets little better salary that a woman
#health is also important
def futuresalary(kid):
    multiplier = 1
    if(kid[1]==1):# if kid is a boy
        multiplier*=1.2
    if(kid[3]==1):# if kid has a rich family
        multiplier*=4
    iq=kid[2]
    health=kid[4]
    return multiplier*(iq*health*1000)

#but its difficult to create such algorithm. but we could use lot of data to train
#a model to predict salary based on data (statistics).
#in this model im using fake data to demostrate possibility of such model.

#calulating outputs
salaries=[futuresalary(k)for k in KIDS]
Y=np.array(salaries, dtype=np.float32)
print("calculated outputs: {0}".format(Y[:4]))
model= LinearRegression()
model.fit(X, Y)

print("prediction columns: name, sex, iq, rich_family, healthy")

def testmodel():
    rk= randomkid()
    pred=model.predict([rk[1:]])/1000
    pred_show= floor(pred[0])
    calc=futuresalary(rk)/1000
    print("kid: {0}, predicted: {1}k, calculated: {2}k".format(rk, pred_show, calc))

for i in range(30):
    testmodel()

'''
RESULTS:

training wit 50 examples:
prediction columns: name, sex, iq, rich_family, healthy
kid: ['John Powell', 0, 15, 1, 75], predicted: 10851k, calculated: 4500.0k
kid: ['Robert Coon', 0, 59, 1, 13], predicted: 6283k, calculated: 3068.0k
kid: ['Jennifer Luten', 1, 76, 0, 8], predicted: -2368k, calculated: 729.6k
kid: ['Peggy Moore', 1, 100, 0, 59], predicted: 7955k, calculated: 7080.0k
kid: ['Cecil Graham', 0, 107, 0, 74], predicted: 11180k, calculated: 7918.0k
kid: ['Elaine Gephart', 1, 89, 1, 12], predicted: 9208k, calculated: 5126.4k
kid: ['Sarah Christensen', 1, 96, 1, 82], predicted: 20548k, calculated: 37785.6k
kid: ['Sharon Drury', 0, 69, 1, 63], predicted: 14928k, calculated: 17388.0k
kid: ['Margaret Mullen', 1, 37, 0, 64], predicted: 1840k, calculated: 2841.6k
kid: ['Dorothy Contreras', 0, 40, 1, 37], predicted: 7837k, calculated: 5920.0k
kid: ['Leslie Peppers', 1, 85, 0, 82], predicted: 9794k, calculated: 8364.0k
kid: ['Emma Gennaro', 1, 23, 0, 26], predicted: -5428k, calculated: 717.6k
kid: ['Ben Koon', 0, 26, 0, 34], predicted: -3697k, calculated: 884.0k
kid: ['Christopher Lamar', 1, 80, 0, 34], predicted: 1997k, calculated: 3264.0k
kid: ['Roy Tucker', 0, 49, 1, 60], predicted: 12293k, calculated: 11760.0k
kid: ['James Dougan', 0, 57, 1, 80], predicted: 16187k, calculated: 18240.0k
kid: ['Mary Coats', 1, 114, 1, 96], predicted: 24626k, calculated: 52531.2k
kid: ['Carroll Rose', 1, 42, 1, 87], predicted: 15414k, calculated: 17539.2k
kid: ['Lisa Holcomb', 0, 99, 0, 38], predicted: 4868k, calculated: 3762.0k
kid: ['Cheryl Mcneal', 1, 63, 1, 24], predicted: 8186k, calculated: 7257.6k
kid: ['Hector Zelinsky', 0, 55, 0, 77], predicted: 5962k, calculated: 4235.0k
kid: ['Colin Arrington', 0, 76, 0, 46], predicted: 3569k, calculated: 3496.0k
kid: ['Simon Polk', 0, 57, 0, 33], predicted: -468k, calculated: 1881.0k
kid: ['Laurie Bailey', 1, 109, 0, 86], predicted: 13016k, calculated: 11248.8k
kid: ['Harvey Henderson', 0, 26, 0, 75], predicted: 2497k, calculated: 1950.0k
kid: ['Norma Ellis', 0, 21, 1, 63], predicted: 9693k, calculated: 5292.0k
kid: ['Thomas Lewis', 0, 110, 0, 45], predicted: 7126k, calculated: 4950.0k
kid: ['Helen Vaneck', 1, 65, 1, 6], predicted: 5684k, calculated: 1872.0k
kid: ['Kari Silva', 1, 52, 0, 81], predicted: 6044k, calculated: 5054.4k
kid: ['Bryant Denney', 0, 70, 0, 19], predicted: -1165k, calculated: 1330.0k

training with 400 examples:
prediction columns: name, sex, iq, rich_family, healthy
kid: ['Celeste Lopez', 0, 88, 1, 9], predicted: 9772k, calculated: 3168.0k
kid: ['Mary Rinehart', 0, 67, 1, 20], predicted: 8803k, calculated: 5360.0k
kid: ['Bertha Young', 0, 29, 0, 20], predicted: -8825k, calculated: 580.0k
kid: ['William Mangan', 0, 108, 0, 75], predicted: 13303k, calculated: 8100.0k
kid: ['William Habegger', 0, 114, 0, 7], predicted: 1142k, calculated: 798.0k
kid: ['Jerrica Duval', 0, 18, 0, 8], predicted: -12739k, calculated: 144.0k
kid: ['Scott Goss', 1, 69, 1, 83], predicted: 23216k, calculated: 27489.6k
kid: ['Kevin Brown', 1, 100, 1, 98], predicted: 30637k, calculated: 47040.0k
kid: ['Roy Hanline', 0, 85, 0, 85], predicted: 11849k, calculated: 7225.0k
kid: ['Veronica Elmore', 0, 105, 0, 89], predicted: 15548k, calculated: 9345.0k
kid: ['Brenda Parkman', 1, 51, 0, 1], predicted: -7206k, calculated: 61.2k
kid: ['Linda Veshedsky', 0, 49, 0, 34], predicted: -3209k, calculated: 1666.0k
kid: ['Ray Jones', 0, 18, 1, 97], predicted: 16387k, calculated: 6984.0k
kid: ['Cristin Hitchcock', 0, 37, 1, 72], predicted: 14378k, calculated: 10656.0k
kid: ['Linda Howry', 1, 23, 0, 15], predicted: -8626k, calculated: 414.0k
kid: ['Amber Morsbach', 0, 32, 0, 69], predicted: 1011k, calculated: 2208.0k
kid: ['Tammy Neal', 0, 28, 1, 68], predicted: 12292k, calculated: 7616.0k
kid: ['Randy Inlow', 1, 104, 1, 74], predicted: 26621k, calculated: 36940.8k
kid: ['Kristine West', 0, 87, 1, 41], predicted: 15762k, calculated: 14268.0k
kid: ['Marguerite Thomas', 0, 79, 1, 82], predicted: 22453k, calculated: 25912.0k
kid: ['James Willits', 1, 78, 0, 54], predicted: 6916k, calculated: 5054.4k
kid: ['George Foster', 0, 62, 1, 79], predicted: 19385k, calculated: 19592.0k
kid: ['Julian Knight', 0, 90, 0, 92], predicted: 13924k, calculated: 8280.0k
kid: ['Carmen Hernandez', 0, 23, 1, 46], predicted: 7340k, calculated: 4232.0k
kid: ['Edward Gill', 0, 36, 0, 96], predicted: 6776k, calculated: 3456.0k
kid: ['Roger Bunton', 1, 72, 0, 55], predicted: 6228k, calculated: 4752.0k
kid: ['Nicholas Cano', 0, 92, 1, 63], predicted: 20715k, calculated: 23184.0k
kid: ['Holly Stewart', 0, 100, 1, 64], predicted: 22079k, calculated: 25600.0k
kid: ['Marie Laperouse', 1, 26, 1, 2], predicted: 1379k, calculated: 249.6k
kid: ['Victor Markey', 0, 83, 1, 79], predicted: 22464k, calculated: 26228.0k


result after traning the model with 2k examples:
prediction columns: name, sex, iq, rich_family, healthy
kid: ['Robert Wright', 0, 63, 0, 4], predicted: -5614k, calculated: 252.0k
kid: ['Helen Leibowitz', 1, 12, 0, 61], predicted: -953k, calculated: 878.4k
kid: ['Willie Garmon', 0, 51, 0, 22], predicted: -4077k, calculated: 1122.0k
kid: ['John Rhoads', 0, 94, 0, 53], predicted: 7229k, calculated: 4982.0k
kid: ['Bryan Crider', 1, 95, 0, 45], predicted: 7517k, calculated: 5130.0k
kid: ['Heather Pick', 1, 106, 0, 99], predicted: 18521k, calculated: 12592.8k
kid: ['Neil Lee', 0, 33, 1, 67], predicted: 11993k, calculated: 8844.0k
kid: ['Reuben Keaton', 0, 46, 0, 1], predicted: -8454k, calculated: 46.0k
kid: ['Jose Kramer', 1, 57, 0, 89], predicted: 10096k, calculated: 6087.6k
kid: ['Kyle Marshall', 0, 45, 1, 48], predicted: 10280k, calculated: 8640.0k
kid: ['Lavon Velasquez', 0, 84, 0, 71], predicted: 9038k, calculated: 5964.0k
kid: ['Michael Poche', 0, 43, 1, 52], predicted: 10712k, calculated: 8944.0k
kid: ['Curtis Najera', 1, 38, 1, 80], predicted: 16523k, calculated: 14592.0k
kid: ['Prince Miller', 0, 56, 0, 67], predicted: 4526k, calculated: 3752.0k
kid: ['David Garrow', 0, 86, 0, 5], predicted: -2310k, calculated: 430.0k
kid: ['Charlene Yamamoto', 0, 87, 0, 7], predicted: -1822k, calculated: 609.0k
kid: ['David Jamison', 0, 89, 1, 59], predicted: 18200k, calculated: 21004.0k
kid: ['Deloris Lott', 0, 26, 1, 66], predicted: 10865k, calculated: 6864.0k
kid: ['Susan Gualtieri', 1, 80, 0, 49], predicted: 6182k, calculated: 4704.0k
kid: ['Christopher Burley', 0, 99, 0, 87], predicted: 13895k, calculated: 8613.0k
kid: ['Martha Williams', 1, 63, 1, 26], predicted: 10415k, calculated: 7862.4k
kid: ['Kathie Seitz', 1, 23, 1, 85], predicted: 15363k, calculated: 9384.0k
kid: ['Robert Gurwell', 0, 45, 0, 36], predicted: -2428k, calculated: 1620.0k
kid: ['Tracy Angles', 0, 15, 0, 48], predicted: -4395k, calculated: 720.0k
kid: ['Veronica Matthews', 0, 109, 0, 56], predicted: 9797k, calculated: 6104.0k
kid: ['Edward Mcguire', 1, 30, 1, 39], predicted: 8216k, calculated: 5616.0k
kid: ['Lucy Tongue', 1, 29, 0, 55], predicted: 302k, calculated: 1914.0k
kid: ['Jason Potts', 1, 102, 0, 37], predicted: 7061k, calculated: 4528.8k
kid: ['James Manson', 1, 100, 1, 81], predicted: 25131k, calculated: 38880.0k
kid: ['Polly Talton', 0, 112, 1, 49], predicted: 19567k, calculated: 21952.0k

results with model trained on 10k examples:
kid: ['Mary Peterson', 0, 72, 1, 25], predicted: 9839k, calculated: 7200.0k
kid: ['Kim Chester', 1, 16, 1, 31], predicted: 4932k, calculated: 2380.8k
kid: ['Joseph Kintner', 0, 88, 1, 63], predicted: 18753k, calculated: 22176.0k
kid: ['Robert Rollman', 0, 115, 0, 38], predicted: 7391k, calculated: 4370.0k
kid: ['Karol Wolfman', 0, 33, 1, 94], predicted: 16924k, calculated: 12408.0k
kid: ['John Hanway', 1, 27, 0, 86], predicted: 5692k, calculated: 2786.4k
kid: ['Katie Mccullough', 1, 28, 1, 21], predicted: 4754k, calculated: 2822.4k
kid: ['Eleanor Baldwin', 0, 81, 1, 43], predicted: 14251k, calculated: 13932.0k
kid: ['Debra Moss', 1, 113, 0, 68], predicted: 13984k, calculated: 9220.8k
kid: ['Wallace Farrell', 0, 41, 1, 27], predicted: 6049k, calculated: 4428.0k
kid: ['Charles Rascon', 1, 56, 0, 78], predicted: 8144k, calculated: 5241.6k
kid: ['Vicky Mellott', 0, 22, 0, 40], predicted: -4691k, calculated: 880.0k
kid: ['Ana Masterson', 0, 79, 1, 77], predicted: 20045k, calculated: 24332.0k
kid: ['Albert Ferdig', 1, 48, 0, 18], predicted: -3622k, calculated: 1036.8k
kid: ['Cory Fico', 0, 82, 0, 16], predicted: -945k, calculated: 1312.0k
kid: ['Jessica Okafor', 1, 73, 1, 69], predicted: 19329k, calculated: 24177.6k
kid: ['Janet Schuster', 1, 29, 0, 9], predicted: -7768k, calculated: 313.2k
kid: ['Patricia Reed', 0, 60, 0, 92], predicted: 9662k, calculated: 5520.0k
kid: ['Coleman Maust', 0, 109, 0, 6], predicted: 883k, calculated: 654.0k
kid: ['Rebecca Garcia', 0, 37, 1, 82], predicted: 15319k, calculated: 12136.0k
kid: ['Jill Champagne', 0, 116, 0, 87], predicted: 16260k, calculated: 10092.0k
kid: ['Barbara Burton', 0, 94, 1, 16], predicted: 11176k, calculated: 6016.0k
kid: ['William Veliz', 1, 81, 1, 32], predicted: 13803k, calculated: 12441.6k
kid: ['Alan Boling', 1, 14, 0, 95], predicted: 5558k, calculated: 1596.0k
kid: ['Meagan Peterson', 1, 86, 0, 74], predicted: 11443k, calculated: 7636.8k
kid: ['Samuel Deshaies', 1, 79, 1, 66], predicted: 19597k, calculated: 25027.2k
kid: ['Nicholas Hou', 1, 106, 1, 46], predicted: 19642k, calculated: 23404.8k
kid: ['Adrian Williams', 1, 46, 0, 58], predicted: 3241k, calculated: 3201.6k
kid: ['Cleveland Willis', 1, 32, 1, 8], predicted: 2971k, calculated: 1228.8k
kid: ['Vita Deleon', 1, 99, 0, 21], predicted: 3733k, calculated: 2494.8k

result for 20k examples:
kid: ['Sarah Jackson', 1, 51, 0, 64], predicted: 4878k, calculated: 3916.8k
kid: ['Linda Napier', 0, 80, 1, 79], predicted: 20913k, calculated: 25280.0k
kid: ['Irene Rea', 0, 36, 1, 80], predicted: 14892k, calculated: 11520.0k
kid: ['Billy Humbert', 0, 84, 0, 6], predicted: -2398k, calculated: 504.0k
kid: ['Scott Yale', 1, 23, 1, 40], predicted: 7440k, calculated: 4416.0k
kid: ['Diane Soloveichik', 0, 107, 0, 65], predicted: 11407k, calculated: 6955.0k
kid: ['Mabel Talley', 0, 112, 1, 53], predicted: 20768k, calculated: 23744.0k
kid: ['Rachel Perez', 1, 15, 1, 49], predicted: 7924k, calculated: 3528.0k
kid: ['Carla Mccoy', 0, 51, 0, 71], predicted: 4590k, calculated: 3621.0k
kid: ['Lauretta Monteith', 0, 53, 0, 51], predicted: 1291k, calculated: 2703.0k
kid: ['John Reeves', 1, 13, 1, 83], predicted: 13730k, calculated: 5179.2k
kid: ['Alene Harrison', 1, 107, 1, 84], predicted: 27155k, calculated: 43142.4k
kid: ['Louise Rader', 0, 27, 0, 28], predicted: -6491k, calculated: 756.0k
kid: ['Julia Jansen', 0, 58, 0, 1], predicted: -6957k, calculated: 58.0k
kid: ['Jody Draper', 1, 68, 1, 12], predicted: 8769k, calculated: 3916.8k
kid: ['Andy Bickley', 1, 107, 0, 76], predicted: 14918k, calculated: 9758.4k
kid: ['Christopher Hines', 1, 36, 1, 32], predicted: 7840k, calculated: 5529.6k
kid: ['Edna Washington', 1, 111, 0, 57], predicted: 12080k, calculated: 7592.4k
kid: ['Christa Higgins', 1, 73, 0, 10], predicted: -1690k, calculated: 876.0k
kid: ['Danielle Garza', 0, 94, 0, 7], predicted: -809k, calculated: 658.0k
kid: ['Gordon Watkins', 0, 83, 0, 81], predicted: 10889k, calculated: 6723.0k
kid: ['James Smith', 0, 111, 1, 86], predicted: 26535k, calculated: 38184.0k
kid: ['Steve Cardin', 0, 36, 0, 72], predicted: 2655k, calculated: 2592.0k
kid: ['Miguel Newson', 0, 22, 0, 90], predicted: 3905k, calculated: 1980.0k
kid: ['Emma Mancine', 0, 94, 1, 78], predicted: 22707k, calculated: 29328.0k
kid: ['Monty Arnold', 0, 65, 0, 86], predicted: 9248k, calculated: 5590.0k
kid: ['Donald Gomez', 0, 73, 0, 67], predicted: 6974k, calculated: 4891.0k
kid: ['Thomas Taylor', 1, 66, 0, 16], predicted: -1602k, calculated: 1267.2k
kid: ['Arnold Cook', 0, 116, 1, 50], predicted: 20795k, calculated: 23200.0k
kid: ['Alberto Cobbs', 0, 69, 0, 1], predicted: -5407k, calculated: 69.0k

'''