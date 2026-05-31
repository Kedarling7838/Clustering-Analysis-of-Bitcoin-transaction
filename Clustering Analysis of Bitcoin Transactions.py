{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "ad4ac14a-970f-4ccf-b3a4-03e0d2dd2213",
   "metadata": {},
   "source": [
    "# “Clustering Analysis of Bitcoin Transactions for Entity Behavior Profiling”"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c7e36bf3-2eee-45aa-ad8b-52f16c6cbfcb",
   "metadata": {},
   "source": [
    "# Business Problem"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9d6a2419-2324-44a7-aa73-d7992347e459",
   "metadata": {},
   "source": [
    "- The Bitcoin network is one of the most widely used cryptocurrency ecosystems, processing millions of transactions daily across diverse entities such as exchanges, miners, traders, holders, and mixers. However, the pseudonymous nature of Bitcoin transactions makes it difficult to distinguish between legitimate financial activity and suspicious or illicit behavior.\n",
    "- By applying clustering techniques to Bitcoin transaction data, we can group entities with similar transactional characteristics, thereby uncovering hidden structures, profiling user behavior, and assisting stakeholders in making data-driven financial and compliance decisions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ee31cbd9-6efe-4d68-b6ae-5f7e5a567d36",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "5f3aceb7-8d56-4684-b2f4-6b35fc7c1c02",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>tx_id</th>\n",
       "      <th>address</th>\n",
       "      <th>entity_type</th>\n",
       "      <th>value_btc</th>\n",
       "      <th>fee_btc</th>\n",
       "      <th>in_degree</th>\n",
       "      <th>out_degree</th>\n",
       "      <th>timestamp</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>tx_1</td>\n",
       "      <td>addr_163</td>\n",
       "      <td>Hodler</td>\n",
       "      <td>2.2354</td>\n",
       "      <td>0.00489</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>2020-01-01 00:00:00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>tx_2</td>\n",
       "      <td>addr_240</td>\n",
       "      <td>Mixer</td>\n",
       "      <td>0.6442</td>\n",
       "      <td>0.00796</td>\n",
       "      <td>18</td>\n",
       "      <td>1</td>\n",
       "      <td>2020-01-01 01:00:00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>tx_3</td>\n",
       "      <td>addr_499</td>\n",
       "      <td>Trader</td>\n",
       "      <td>0.7315</td>\n",
       "      <td>0.00581</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2020-01-01 02:00:00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>tx_4</td>\n",
       "      <td>addr_111</td>\n",
       "      <td>Mixer</td>\n",
       "      <td>3.1117</td>\n",
       "      <td>0.00349</td>\n",
       "      <td>10</td>\n",
       "      <td>13</td>\n",
       "      <td>2020-01-01 03:00:00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>tx_5</td>\n",
       "      <td>addr_16</td>\n",
       "      <td>Mixer</td>\n",
       "      <td>1.1827</td>\n",
       "      <td>0.00812</td>\n",
       "      <td>11</td>\n",
       "      <td>13</td>\n",
       "      <td>2020-01-01 04:00:00</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  tx_id   address entity_type  value_btc  fee_btc  in_degree  out_degree  \\\n",
       "0  tx_1  addr_163      Hodler     2.2354  0.00489          3           4   \n",
       "1  tx_2  addr_240       Mixer     0.6442  0.00796         18           1   \n",
       "2  tx_3  addr_499      Trader     0.7315  0.00581          2           2   \n",
       "3  tx_4  addr_111       Mixer     3.1117  0.00349         10          13   \n",
       "4  tx_5   addr_16       Mixer     1.1827  0.00812         11          13   \n",
       "\n",
       "             timestamp  \n",
       "0  2020-01-01 00:00:00  \n",
       "1  2020-01-01 01:00:00  \n",
       "2  2020-01-01 02:00:00  \n",
       "3  2020-01-01 03:00:00  \n",
       "4  2020-01-01 04:00:00  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=pd.read_csv(\"C:\\\\Users\\\\Asus\\\\OneDrive\\\\Desktop\\\\power shheets\\\\Projects\\\\btc_transactions_sample.csv\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "96ba45bd-ab80-4cc0-98fa-21003d23c161",
   "metadata": {},
   "source": [
    "Dataset Columns\n",
    "\n",
    "- tx_id → Transaction ID- address → Wallet address (synthetic)\n",
    "- entity_type → Type of entity (Exchange, Miner, Trader, Hodler, Mixer)\n",
    "- value_btc → Transaction value in BTC\n",
    "- fee_btc → Transaction fee in BTC\n",
    "- in_degree → Number of unique senders\n",
    "- out_degree → Number of unique receivers\n",
    "- timestamp → Date & time of transaction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "68dad8a5-d3a7-4647-8ad8-3026d91a1ebb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 3000 entries, 0 to 2999\n",
      "Data columns (total 8 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   tx_id        3000 non-null   object \n",
      " 1   address      3000 non-null   object \n",
      " 2   entity_type  3000 non-null   object \n",
      " 3   value_btc    3000 non-null   float64\n",
      " 4   fee_btc      3000 non-null   float64\n",
      " 5   in_degree    3000 non-null   int64  \n",
      " 6   out_degree   3000 non-null   int64  \n",
      " 7   timestamp    3000 non-null   object \n",
      "dtypes: float64(2), int64(2), object(4)\n",
      "memory usage: 187.6+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "1f07ce99-6f82-46e4-85f8-1fff6f56699a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>value_btc</th>\n",
       "      <th>fee_btc</th>\n",
       "      <th>in_degree</th>\n",
       "      <th>out_degree</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>3000.000000</td>\n",
       "      <td>3000.000000</td>\n",
       "      <td>3000.000000</td>\n",
       "      <td>3000.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>1.968918</td>\n",
       "      <td>0.004952</td>\n",
       "      <td>10.229667</td>\n",
       "      <td>10.112333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>1.966096</td>\n",
       "      <td>0.002846</td>\n",
       "      <td>5.419414</td>\n",
       "      <td>5.540814</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000100</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.574650</td>\n",
       "      <td>0.002490</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>5.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>1.341600</td>\n",
       "      <td>0.004900</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>10.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>2.718400</td>\n",
       "      <td>0.007392</td>\n",
       "      <td>15.000000</td>\n",
       "      <td>15.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>15.447100</td>\n",
       "      <td>0.009990</td>\n",
       "      <td>19.000000</td>\n",
       "      <td>19.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         value_btc      fee_btc    in_degree   out_degree\n",
       "count  3000.000000  3000.000000  3000.000000  3000.000000\n",
       "mean      1.968918     0.004952    10.229667    10.112333\n",
       "std       1.966096     0.002846     5.419414     5.540814\n",
       "min       0.000000     0.000100     1.000000     1.000000\n",
       "25%       0.574650     0.002490     6.000000     5.000000\n",
       "50%       1.341600     0.004900    10.000000    10.000000\n",
       "75%       2.718400     0.007392    15.000000    15.000000\n",
       "max      15.447100     0.009990    19.000000    19.000000"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "1f5f4424-ee28-4de9-9a21-97976b1842f9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Hodler', 'Mixer', 'Trader', 'Miner', 'Exchange'], dtype=object)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['entity_type'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "b7d55fbf-e359-4b39-918d-5c7c23824801",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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sFC333677rrrLn366ae66667dNttt+njjz/uaHter2/fvpKkV199VcOGDdOCBQu0fPlyLViwQMOGDdOKFSuc6gAA8AYdDiwpKSl6+umnNWPGjAuW2e12ZWdn68knn9SMGTM0atQovfXWWzp16pTefvvtNj8zOztbN910kzIzMxUTE6PMzExNmTJF2dnZHW3P6/Xr10+StG/fPp0+fdppDMvp06e1b98+pzoAALyBSwfdlpWVqaKiQlOnTnXMs1qtmjx5cruDQAsLC53WkaSbb76ZgaOtaL5KKDg4WP7+/rr//vsVGRmp+++/XwEBAQoODpafnx9XCQEAvIpLB91WVFRIksLCwpzmh4WF6euvv253vdbWaf681jQ0NDg9O8dms11Oy6bT8iqh6upqp2WHDx92qktKSurc5oAuILynRQFVX0rHve8iy4CqLxXekye9t8T+Ng63XCVksTj/B7Db7RfMu9J1srKytHjx4stv0qRa3sHWYrHIbrc7pn18fNTU1HRBHQDX+c2Eboot+I1U4OlOXC9W578f/oH9bRwuDSzh4eGSzh8xaXmn1crKyguOoPx4vR8fTbnYOpmZmcrIyHBM22w2RUVFXW7rptG7d29JUs+ePdW7d2+nI1dRUVH6/vvvVVtb66gD4Fp/Kjqj2/91lWJjYjzdissVl5ToT0tn6ReebsRA2N/G4dLAMmjQIIWHh2vr1q0aN26cJOnMmTPasWOHnn322TbXi4+P19atW7VgwQLHvC1btrQ7DsNqtcpqtbqueZM4cOCAJKm2tlaNjY1OyyorK1VfX++o+/G4IABXrqLWrvpew6TIn3i6FZerr2hSRa394oVdCPvbODocWGpra3Xo0CHHdFlZmfbv36+QkBANGDBAjz76qJ555hkNHTpUQ4cO1TPPPKPu3btr1qxZjnVmz56tfv36KSsrS5L0yCOP6IYbbtCzzz6rW2+9Ve+99562bdumXbt2ueArepevvvrK8b5nz5564IEHNHjwYH311VdavXq1I7C0rAMAwOw6HFj27Nmj5ORkx3TzaZk5c+Zo1apVevzxx1VfX68HHnhAP/zwg66//npt2bJFgYGBjnWOHDkiH59/DGBKSEjQunXr9Ic//EFPPfWUhgwZonfeeUfXX3/9lXw3r9Q8RqVXr146efKkli5d6ljm5+enXr16qaqqylEHAIA36HBg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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "contineous=['value_btc','fee_btc','in_degree','out_degree']\n",
    "plt.boxplot(df[contineous])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "86de81a9-6758-4256-9e5a-9c9229ca6731",
   "metadata": {},
   "source": [
    "- value_btc feature has outliers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "2434a409-1589-43c1-95b1-037ecb0f90e4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "value_btc     1.893495\n",
       "fee_btc       0.049389\n",
       "in_degree    -0.059642\n",
       "out_degree   -0.028539\n",
       "dtype: float64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[contineous].skew()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "717f4467-e725-45e3-b5d2-4ef17cbdebc9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tx_id          0\n",
       "address        0\n",
       "entity_type    0\n",
       "value_btc      0\n",
       "fee_btc        0\n",
       "in_degree      0\n",
       "out_degree     0\n",
       "timestamp      0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "0b30e589-d99a-4bfa-a06e-8b39fb688efa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "np.int64(0)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.duplicated().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dacd6803-9b18-4e38-9d1b-9f121d4499d8",
   "metadata": {},
   "source": [
    "# Data Cleaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "5dc1daf9-9543-4393-8bb6-d300623bd660",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>entity_type</th>\n",
       "      <th>value_btc</th>\n",
       "      <th>fee_btc</th>\n",
       "      <th>in_degree</th>\n",
       "      <th>out_degree</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Hodler</td>\n",
       "      <td>2.2354</td>\n",
       "      <td>0.00489</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Mixer</td>\n",
       "      <td>0.6442</td>\n",
       "      <td>0.00796</td>\n",
       "      <td>18</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Trader</td>\n",
       "      <td>0.7315</td>\n",
       "      <td>0.00581</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Mixer</td>\n",
       "      <td>3.1117</td>\n",
       "      <td>0.00349</td>\n",
       "      <td>10</td>\n",
       "      <td>13</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Mixer</td>\n",
       "      <td>1.1827</td>\n",
       "      <td>0.00812</td>\n",
       "      <td>11</td>\n",
       "      <td>13</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  entity_type  value_btc  fee_btc  in_degree  out_degree\n",
       "0      Hodler     2.2354  0.00489          3           4\n",
       "1       Mixer     0.6442  0.00796         18           1\n",
       "2      Trader     0.7315  0.00581          2           2\n",
       "3       Mixer     3.1117  0.00349         10          13\n",
       "4       Mixer     1.1827  0.00812         11          13"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Removing unnecessory features\n",
    "df=df.drop(['tx_id','address','timestamp'],axis=1)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "22963911-b22a-4ef2-886f-4ea83488a34f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Q1: 0.57465\n",
      "Q3: 2.7184\n",
      "IQR: 2.14375\n"
     ]
    }
   ],
   "source": [
    "# Removing Outliers\n",
    "Q1=df['value_btc'].quantile(0.25)\n",
    "print(\"Q1:\",Q1)\n",
    "Q3=df['value_btc'].quantile(0.75)\n",
    "print(\"Q3:\",Q3)\n",
    "IQR = Q3-Q1\n",
    "print(\"IQR:\",IQR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "94527691-4d70-4ff6-8eb3-e6236dff391f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "upper_limit 5.934025\n",
      "lower_limit -2.6409749999999996\n"
     ]
    }
   ],
   "source": [
    "upper_limit=Q3+(1.5*IQR)\n",
    "print(\"upper_limit\",upper_limit)\n",
    "lower_limit=Q1-(1.5*IQR)\n",
    "print(\"lower_limit\",lower_limit)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "4be1c924-b910-4251-85ea-e389d3a84e89",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>entity_type</th>\n",
       "      <th>value_btc</th>\n",
       "      <th>fee_btc</th>\n",
       "      <th>in_degree</th>\n",
       "      <th>out_degree</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>34</th>\n",
       "      <td>Trader</td>\n",
       "      <td>8.8517</td>\n",
       "      <td>0.00328</td>\n",
       "      <td>10</td>\n",
       "      <td>19</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>41</th>\n",
       "      <td>Exchange</td>\n",
       "      <td>6.6494</td>\n",
       "      <td>0.00392</td>\n",
       "      <td>18</td>\n",
       "      <td>12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>53</th>\n",
       "      <td>Mixer</td>\n",
       "      <td>6.2434</td>\n",
       "      <td>0.00249</td>\n",
       "      <td>9</td>\n",
       "      <td>10</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>58</th>\n",
       "      <td>Hodler</td>\n",
       "      <td>8.2720</td>\n",
       "      <td>0.00680</td>\n",
       "      <td>6</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>60</th>\n",
       "      <td>Miner</td>\n",
       "      <td>6.4953</td>\n",
       "      <td>0.00600</td>\n",
       "      <td>8</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2960</th>\n",
       "      <td>Miner</td>\n",
       "      <td>6.7959</td>\n",
       "      <td>0.00788</td>\n",
       "      <td>15</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2970</th>\n",
       "      <td>Trader</td>\n",
       "      <td>6.4754</td>\n",
       "      <td>0.00968</td>\n",
       "      <td>13</td>\n",
       "      <td>14</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2977</th>\n",
       "      <td>Mixer</td>\n",
       "      <td>10.7728</td>\n",
       "      <td>0.00059</td>\n",
       "      <td>7</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2983</th>\n",
       "      <td>Exchange</td>\n",
       "      <td>7.5846</td>\n",
       "      <td>0.00536</td>\n",
       "      <td>19</td>\n",
       "      <td>12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2991</th>\n",
       "      <td>Mixer</td>\n",
       "      <td>9.2682</td>\n",
       "      <td>0.00766</td>\n",
       "      <td>19</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>159 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     entity_type  value_btc  fee_btc  in_degree  out_degree\n",
       "34        Trader     8.8517  0.00328         10          19\n",
       "41      Exchange     6.6494  0.00392         18          12\n",
       "53         Mixer     6.2434  0.00249          9          10\n",
       "58        Hodler     8.2720  0.00680          6           4\n",
       "60         Miner     6.4953  0.00600          8           8\n",
       "...          ...        ...      ...        ...         ...\n",
       "2960       Miner     6.7959  0.00788         15           1\n",
       "2970      Trader     6.4754  0.00968         13          14\n",
       "2977       Mixer    10.7728  0.00059          7           2\n",
       "2983    Exchange     7.5846  0.00536         19          12\n",
       "2991       Mixer     9.2682  0.00766         19           2\n",
       "\n",
       "[159 rows x 5 columns]"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[(df['value_btc']<lower_limit)|(df['value_btc']>upper_limit)]                  #Outlier records"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "37f99c54-ff6f-4ffd-a5a6-a696e884d312",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0       2.2354\n",
       "1       0.6442\n",
       "2       0.7315\n",
       "3       3.1117\n",
       "4       1.1827\n",
       "         ...  \n",
       "2995    0.3738\n",
       "2996    0.9707\n",
       "2997    0.8522\n",
       "2998    1.1306\n",
       "2999    3.0445\n",
       "Name: value_btc, Length: 3000, dtype: float64"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['value_btc']=df['value_btc'].clip(lower=-2.6409749999999996,upper=5.934025)\n",
    "df['value_btc']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "8544fb5a-ef29-4846-809d-a9b42439faaf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAhYAAAGdCAYAAABO2DpVAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAE05JREFUeJzt3W9sXQXdwPFf2cKlhfbiwMGWXbaFDFuYA9sR3aaGKZI0skdeqOhDyWLEiJkgLkadJo9CotUXj+HF4sIWAyGLjhgz5IWMzBdjJjjdOheJtmzIFqpjEoj2dn+8xHGfF2Z9rKzAbX+3t3f7fJKT5ZyeP7/wpl/OPb2npVqtVgMAIMEFjR4AADh3CAsAII2wAADSCAsAII2wAADSCAsAII2wAADSCAsAIM3s6b7g66+/HkePHo329vZoaWmZ7ssDAJNQrVZjdHQ05s+fHxdcMPF9iWkPi6NHj0apVJruywIACYaHh2PBggUT/nzaw6K9vT0i/jVYR0fHdF8eAJiEcrkcpVJp7Pf4RKY9LM58/NHR0SEsAKDJvNVjDB7eBADSCAsAII2wAADSCAsAII2wAADSCAsAIE3NYfGXv/wl+vr64rLLLou2tra44YYbYmBgoB6zAQBNpqbvsfjb3/4Wq1atitWrV8eTTz4Zc+fOjT/96U9x6aWX1mk8AKCZ1BQW3//+96NUKsXDDz88tm3RokXZMwEATaqmj0KeeOKJWL58eXziE5+IuXPnxnve857YsmXLmx5TqVSiXC6PWwCAc1NNYfHCCy/Epk2bYsmSJfHUU0/F3XffHffee288+uijEx7T398fxWJxbPECMgA4d7VUq9Xq2935wgsvjOXLl8czzzwztu3ee++NvXv3xq9//euzHlOpVKJSqYytn3mJycjIiHeFAECTKJfLUSwW3/L3d03PWMybNy+uvfbacdu6urriZz/72YTHFAqFKBQKtVwGmEYnT56MoaGhlHOdOnUqjhw5EosWLYrW1tYpn6+zszPa2toSJgOmS01hsWrVqnjuuefGbTt48GAsXLgwdShg+gwNDUVPT0+jxzirgYGB6O7ubvQYQA1qCosvf/nLsXLlyvjud78bn/zkJ+O3v/1tbN68OTZv3lyv+YA66+zsTPsumsHBwejr64utW7dGV1fXlM/X2dmZMBUwnWoKixtvvDG2b98eGzZsiAceeCAWL14cDz74YNxxxx31mg+os7a2tvS7Al1dXe40wHmqprCIiLj11lvj1ltvrccsAECT864QACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACCNsAAA0ggLACBNTWHx7W9/O1paWsYtV155Zb1mAwCazOxaD7juuuvil7/85dj6rFmzUgcCAJpXzWExe/ZsdykAgLOq+RmLQ4cOxfz582Px4sXxqU99Kl544YU33b9SqUS5XB63AADnpprC4r3vfW88+uij8dRTT8WWLVvi2LFjsXLlynj11VcnPKa/vz+KxeLYUiqVpjw0ADAztVSr1epkDz5x4kRcffXV8dWvfjXWr19/1n0qlUpUKpWx9XK5HKVSKUZGRqKjo2OylwZmoP3790dPT08MDAxEd3d3o8cBEpXL5SgWi2/5+7vmZyz+3cUXXxzvfve749ChQxPuUygUolAoTOUyAECTmNL3WFQqlRgcHIx58+ZlzQMANLGawuIrX/lKPP3003H48OH4zW9+Ex//+MejXC7H2rVr6zUfANBEavoo5M9//nN8+tOfjldeeSXe+c53xvve977Ys2dPLFy4sF7zAQBNpKaw2LZtW73mAADOAd4VAgCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQBphAQCkERYAQJophUV/f3+0tLTEfffdlzQOANDMJh0We/fujc2bN8eyZcsy5wEAmtikwuL48eNxxx13xJYtW+Id73hH9kwAQJOaVFisW7cuPvrRj8bN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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.boxplot(df['value_btc'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2ec0cdfa-a748-4852-a21e-f6a913992de9",
   "metadata": {},
   "source": [
    "# Data Wrangling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "45fd7a53-a478-4e05-8286-cb2911d8dbcc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>entity_type</th>\n",
       "      <th>value_btc</th>\n",
       "      <th>fee_btc</th>\n",
       "      <th>in_degree</th>\n",
       "      <th>out_degree</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.0</td>\n",
       "      <td>2.2354</td>\n",
       "      <td>0.00489</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3.0</td>\n",
       "      <td>0.6442</td>\n",
       "      <td>0.00796</td>\n",
       "      <td>18</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   entity_type  value_btc  fee_btc  in_degree  out_degree\n",
       "0          1.0     2.2354  0.00489          3           4\n",
       "1          3.0     0.6442  0.00796         18           1"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import OrdinalEncoder\n",
    "oe=OrdinalEncoder()\n",
    "df['entity_type']=oe.fit_transform(df[['entity_type']])\n",
    "df.head(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c026d772-e5a6-407b-86e1-47e73e4ec3f5",
   "metadata": {},
   "source": [
    "# 1. K-Means Algoritham"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "be28d601-d7e4-4ea7-9196-bfe727401c6f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[194380.95100619618, 124715.18699292245, 85556.21868304741, 57884.46546163903, 52124.50225183372, 45177.526104079414, 41486.8970272618, 37110.58052257861, 34205.801176849236, 32385.261284615895]\n"
     ]
    }
   ],
   "source": [
    "wcss=[]\n",
    "for k in range(1,11):\n",
    "    from sklearn.cluster import KMeans\n",
    "    kmeans=KMeans(n_clusters= k, init='k-means++')\n",
    "    kmeans.fit(df)\n",
    "    wcss.append(kmeans.inertia_)\n",
    "print(wcss)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "d6f102a2-91fc-42e2-9c68-acac3742e767",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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fYM63nAvnGNFRERE13VpiFIoFHYTqoDa3nzdHeDr7nzNuqvNFds6a8RVT2neLLwzJBEREbUKdRccynkHII5YERERUYvWlLliNwvnWDUzzrEiIiKyvZs9V4xzrIiIiKjNsJe5YpxjRURERGQjDFZERERENsJgRURERGQjDFZERERENsJgRURERGQjDFZERERENsJgRURERGQjDFZERERENsJgRURERGQjDFZERERENsJb2jSzulszFhcXy9wJERERNVTd7/b1brHMYNXMSkpKAACdO3eWuRMiIiJqrJKSEmi12qvuV4jrRS+yKYvFgpycHLi5uUGhUNjsfYuLi9G5c2dkZWVd867b9o7HYV94HPantRwLj8O+8DiuTwiBkpIS6PV6KJVXn0nFEatmplQq0alTp5v2/u7u7i36H4o6PA77wuOwP63lWHgc9oXHcW3XGqmqw8nrRERERDbCYEVERERkIwxWrYRarcbixYuhVqvlbuWG8DjsC4/D/rSWY+Fx2Bceh+1w8joRERGRjXDEioiIiMhGGKyIiIiIbITBioiIiMhGGKyIiIiIbITBqoXbv38/Jk6cCL1eD4VCga1bt8rdUpNERkZiyJAhcHNzg4+PD+677z6cPHlS7rYa7f3330f//v2lxemGDx+OH374Qe62blhkZCQUCgXmzp0rdyuNsmTJEigUCquHTqeTu60myc7OxqOPPgovLy+4uLhg4MCBSEpKkrutRunatWu9/z0UCgVmzZold2uNUlNTg7///e/o1q0bNBoNunfvjqVLl8JiscjdWqOVlJRg7ty58Pf3h0ajQUhICBITE+Vu67qu99snhMCSJUug1+uh0WgwatQoHDt2rFl6Y7Bq4crKyjBgwACsXbtW7lZuyL59+zBr1izEx8dj586dqKmpQVhYGMrKyuRurVE6deqE5cuX49dff8Wvv/6Ku+66C/fee2+z/QN9MyQmJmLDhg3o37+/3K00Sb9+/ZCbmys9UlJS5G6p0QwGA0aMGAEnJyf88MMPOH78ON5++220b99e7tYaJTEx0ep/i507dwIAHnzwQZk7a5wVK1Zg/fr1WLt2LU6cOIGVK1firbfewpo1a+RurdGefPJJ7Ny5E5999hlSUlIQFhaGMWPGIDs7W+7Wrul6v30rV67E6tWrsXbtWiQmJkKn02Hs2LHS/XpvKkGtBgCxZcsW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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(range(1,11),wcss,marker='*')\n",
    "plt.xticks(range(1,11))\n",
    "plt.title('Elbow Method')\n",
    "plt.xlabel(\"Number of clusters\")\n",
    "plt.ylabel(\"wcss\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "b87671ba-1e23-4358-bf69-69255500f2a8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>entity_type</th>\n",
       "      <th>value_btc</th>\n",
       "      <th>fee_btc</th>\n",
       "      <th>in_degree</th>\n",
       "      <th>out_degree</th>\n",
       "      <th>km_clusters</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.0</td>\n",
       "      <td>2.2354</td>\n",
       "      <td>0.00489</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3.0</td>\n",
       "      <td>0.6442</td>\n",
       "      <td>0.00796</td>\n",
       "      <td>18</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.0</td>\n",
       "      <td>0.7315</td>\n",
       "      <td>0.00581</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3.0</td>\n",
       "      <td>3.1117</td>\n",
       "      <td>0.00349</td>\n",
       "      <td>10</td>\n",
       "      <td>13</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.0</td>\n",
       "      <td>1.1827</td>\n",
       "      <td>0.00812</td>\n",
       "      <td>11</td>\n",
       "      <td>13</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2995</th>\n",
       "      <td>1.0</td>\n",
       "      <td>0.3738</td>\n",
       "      <td>0.00122</td>\n",
       "      <td>16</td>\n",
       "      <td>8</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2996</th>\n",
       "      <td>3.0</td>\n",
       "      <td>0.9707</td>\n",
       "      <td>0.00165</td>\n",
       "      <td>3</td>\n",
       "      <td>15</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2997</th>\n",
       "      <td>4.0</td>\n",
       "      <td>0.8522</td>\n",
       "      <td>0.00443</td>\n",
       "      <td>12</td>\n",
       "      <td>13</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2998</th>\n",
       "      <td>2.0</td>\n",
       "      <td>1.1306</td>\n",
       "      <td>0.00403</td>\n",
       "      <td>19</td>\n",
       "      <td>8</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2999</th>\n",
       "      <td>1.0</td>\n",
       "      <td>3.0445</td>\n",
       "      <td>0.00746</td>\n",
       "      <td>19</td>\n",
       "      <td>14</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>3000 rows × 6 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      entity_type  value_btc  fee_btc  in_degree  out_degree  km_clusters\n",
       "0             1.0     2.2354  0.00489          3           4            1\n",
       "1             3.0     0.6442  0.00796         18           1            3\n",
       "2             4.0     0.7315  0.00581          2           2            1\n",
       "3             3.0     3.1117  0.00349         10          13            4\n",
       "4             3.0     1.1827  0.00812         11          13            4\n",
       "...           ...        ...      ...        ...         ...          ...\n",
       "2995          1.0     0.3738  0.00122         16           8            3\n",
       "2996          3.0     0.9707  0.00165          3          15            0\n",
       "2997          4.0     0.8522  0.00443         12          13            4\n",
       "2998          2.0     1.1306  0.00403         19           8            3\n",
       "2999          1.0     3.0445  0.00746         19          14            2\n",
       "\n",
       "[3000 rows x 6 columns]"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.cluster import KMeans\n",
    "kmeans=KMeans(n_clusters= 5, init='k-means++',random_state=True)\n",
    "df['km_clusters']=kmeans.fit_predict(df)\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "28ff34a1-3c88-42aa-824d-d8b794c9f6b3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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vb0dK1euHM8UiWgS8SRoIpqRNm7cCBcXF6SmpqKhoQHHjx9Hbm6uRcy2bdvQ0dGBDRs2oKqqCs3NzTh16hRSUlIwODgIX19fJCcn45133kFJSQnq6+uRkpICFxcXq71CQ+3evRu//PILtm3bhpqaGphMJhQXF2P79u0AgAULFiAxMRGbNm1CUVERWlpacPHiReTk5OD48eOT9roQ0b1hAUREM5KPjw9++uknNDQ0IDw8HO+//77Voa6AgACUl5djcHAQa9asQWhoKHbu3AmNRgMXlztff/v27YPBYEB8fDxWr16NqKgoPProo/Dw8Bh1+4sXL8a5c+dgMpmwcuVKhIeHIysrC1qtVokpKCjApk2bkJGRAb1ej7Vr1+LChQvK3iYimjoquZcD6URETqKnpweBgYHIy8tDamrqVA+HiCYJzwEiIqdmNBpx+fJlLFmyBLdu3cKHH34IAHjhhRemeGRENJl4CIyIpqW9e/fCx8dn2EdcXNyEbis3NxdhYWFYvXo1enp6UFZWhoceemhCt0FE0wsPgRHRtNTR0YGOjo5h13l6eiIwMNDOIyIiR8ICiIiIiJwOD4ERERGR02EBRERERE6HBRARERE5HRZARERE5HRYABEREZHTYQFERERETocFEBERETkdFkBERETkdP4HCqchsL/xGy4AAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.scatterplot(x='in_degree',y='out_degree',hue='km_clusters',data=df)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "df0a06e2-fbbc-4c22-9257-3027a9cd2b7a",
   "metadata": {},
   "source": [
    "# 2.Heirachical Clustering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "9cdb1b95-2f39-466a-8322-d74eccd798d4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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7777LrRaLSZNmoSCggL07NkT27ZtQ0BAgGkdixcvhre3N8aNGwetVouBAwdixYoVkMvl7totIiIi8lAyQRAEdzfC3YqLixEUFISioiIEBga6uznNXnllNRL//BMA4NzHQ+Hr4/Yx9kSSxt9Jaqrqc/72uDE+RERERK7C4ENERESSweBDREREksHgQ0RERJLB4ENERESSweBDREREksHgQ0RERJLB4ENERESSweBDREREksHgQ0RERJLB4ENERESSweBDREREksHgQ0RERJLB4ENERESSweBDREREksHgQ0RERJLB4ENERESSweBDREREksHgQ0RERJLB4ENERESS4e3uBhAReQJBEKCt0ru7GW5RXllt87HUqBVyyGQydzeDXIzBh4gkTxAEPPn5IRzLKHB3U9wuec5OdzfBbZJjgvHt670Zfpo5dnURkeRpq/QMPYSjGQWSrfpJCSs+RERmjv5pEHx95O5uBjWi8ko9kufscHczqJEw+BARmfH1kcPXh38aiZordnURERGRZDD4EBERkWQw+BAREZFkMPgQERGRZDD4EBERkWQw+BAREZFkMPgQERGRZDQo+FRUVDirHUREREQuJzr4GAwG/OUvf0GrVq3g7++P3377DQDwwQcfYNmyZU5vIBEREZGziA4+c+bMwYoVK7BgwQL4+PiYpnfu3Bn//ve/ndo4IiIiImcSHXxWrVqFL7/8Es8++yzk8nvfZ9OlSxdcuHDBqY0jIiIicibRX0hz48YNxMfH15huMBhQVVXllEYRAEEAqsrd3QrXqDT79uPKcgDN9AshFb6ATObuVhARkRnRwadTp07Yt28fYmJiLKZ/++236Natm6h1paamYtOmTbhw4QLUajX69OmD+fPno0OHDqZlJk6ciJUrV1q8rmfPnjh8+LDpuU6nw4wZM7Bu3TpotVoMHDgQn332GVq3bi129zyDIABfDQUyf3F3S1xDUAJYfvfx/4sHZDq3NsdlonsBL25l+CEi8iCig8+HH36I559/Hjdu3IDBYMCmTZtw8eJFrFq1Cj/88IOode3ZsweTJ0/GAw88gOrqasyaNQtDhgzBuXPn4OfnZ1pu2LBhWL58uem5+dgiAJg6dSq+//57rF+/HqGhoZg+fTpGjBiBY8eOWXTHNRlV5c039ADwlemQrhrv7ma4Xubhu++lj1/dyxIRUaMQHXxG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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import scipy.cluster.hierarchy as sch\n",
    "data=sch.linkage(df,method='ward')\n",
    "dendrogram=sch.dendrogram(data)\n",
    "plt.xlabel(\"value_btc\")\n",
    "plt.ylabel(\"Eucledian Distance\")\n",
    "plt.title(\"Dendrogram\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "4bc12d53-307f-4d29-b2a0-599241ab6595",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>entity_type</th>\n",
       "      <th>value_btc</th>\n",
       "      <th>fee_btc</th>\n",
       "      <th>in_degree</th>\n",
       "      <th>out_degree</th>\n",
       "      <th>km_clusters</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "      <td>0.00489</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.00796</td>\n",
       "      <td>18</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.0</td>\n",
       "      <td>2</td>\n",
       "      <td>0.00581</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.00349</td>\n",
       "      <td>10</td>\n",
       "      <td>13</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.00812</td>\n",
       "      <td>11</td>\n",
       "      <td>13</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2995</th>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.00122</td>\n",
       "      <td>16</td>\n",
       "      <td>8</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2996</th>\n",
       "      <td>3.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0.00165</td>\n",
       "      <td>3</td>\n",
       "      <td>15</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2997</th>\n",
       "      <td>4.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.00443</td>\n",
       "      <td>12</td>\n",
       "      <td>13</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2998</th>\n",
       "      <td>2.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.00403</td>\n",
       "      <td>19</td>\n",
       "      <td>8</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2999</th>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>0.00746</td>\n",
       "      <td>19</td>\n",
       "      <td>14</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>3000 rows × 6 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      entity_type  value_btc  fee_btc  in_degree  out_degree  km_clusters\n",
       "0             1.0          2  0.00489          3           4            1\n",
       "1             3.0          0  0.00796         18           1            3\n",
       "2             4.0          2  0.00581          2           2            1\n",
       "3             3.0          0  0.00349         10          13            4\n",
       "4             3.0          0  0.00812         11          13            4\n",
       "...           ...        ...      ...        ...         ...          ...\n",
       "2995          1.0          0  0.00122         16           8            3\n",
       "2996          3.0          1  0.00165          3          15            0\n",
       "2997          4.0          0  0.00443         12          13            4\n",
       "2998          2.0          0  0.00403         19           8            3\n",
       "2999          1.0          3  0.00746         19          14            2\n",
       "\n",
       "[3000 rows x 6 columns]"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.cluster import AgglomerativeClustering\n",
    "hc=AgglomerativeClustering(n_clusters=4,linkage='ward')\n",
    "df[\"value_btc\"]= hc.fit_predict(df)\n",
    "df"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
