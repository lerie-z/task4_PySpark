import pyspark
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import desc, count, sum, rank, row_number, lower


# function to load table from database
def read_table(spark: SparkSession, table: str):
    df = spark.read \
        .format("jdbc") \
        .option("url", "jdbc:postgresql://localhost:5432/pagila") \
        .option("dbtable", table) \
        .option("user", "postgres") \
        .option("password", "1111") \
        .option("driver", "org.postgresql.Driver")
    return df.load()


# amount of movies in each category, descending
def query_task1(spark: SparkSession):
    df_cat = read_table(spark, 'category')
    df_film_cat = read_table(spark, 'film_category')

    df = df_film_cat.join(df_cat, df_film_cat['category_id'] == df_cat['category_id'], 'inner')
    df = df.select(df['name'].alias('category_name')).groupBy('category_name').count().sort(desc('count'))
    return df.select(df['category_name'], df['count'].alias('film_count'))


# 10 actors with most rented films, descending
def query_task2(spark: SparkSession):
    df_rental = read_table(spark, 'rental')
    df_inventory = read_table(spark, 'inventory')
    df_film_actor = read_table(spark, 'film_actor')
    df_actor = read_table(spark, 'actor')

    df = df_rental.join(df_inventory, df_rental['inventory_id'] == df_inventory['inventory_id'], 'inner') \
                  .join(df_film_actor, df_inventory['film_id'] == df_film_actor['film_id'])
    df = df.groupBy(df['actor_id']).agg(count(df['actor_id']).alias('total'))
    df = df.join(df_actor, df['actor_id'] == df_actor['actor_id'], 'inner').sort(desc('total'))
    return df.select(df['first_name'], df['last_name']).limit(10)


# film category people spent the most money on
def query_task3(spark: SparkSession):
    df_rental = read_table(spark, 'rental')
    df_cat = read_table(spark, 'category')
    df_inventory = read_table(spark, 'inventory')
    df_payment = read_table(spark, 'payment')
    df_film_cat = read_table(spark, 'film_category')

    df = df_payment.join(df_rental, df_payment['rental_id'] == df_rental['rental_id'], 'inner') \
                   .join(df_inventory, df_rental['inventory_id'] == df_inventory['inventory_id'], 'inner') \
                   .join(df_film_cat, df_inventory['film_id'] == df_film_cat['film_id'], 'inner')
    df = df.groupBy(df['category_id']).agg(sum(df['amount']).alias('amount'))
    df = df.join(df_cat, df_film_cat['category_id'] == df_cat['category_id']).sort(desc('amount'))
    return df.select(df['name']).limit(1)


# films that are not in INVENTORY(without IN operator)
def query_task4(spark: SparkSession):
    df_inventory = read_table(spark, 'inventory')
    df_film = read_table(spark, 'film')

    df = df_film.join(df_inventory, df_inventory['film_id'] == df_film['film_id'], 'left')
    return df.filter(df_inventory['film_id'].isNull()).select(df['title'])


# top 3 actors who appeared most frequently in CHILDREN category
# (case when the amount of movies is the same, show all)
def query_task5(spark: SparkSession):
    df_film_cat = read_table(spark, 'film_category')
    df_cat = read_table(spark, 'category')
    df_film_actor = read_table(spark, 'film_actor')
    df_actor = read_table(spark, 'actor')
    df_film = read_table(spark, 'film')

    df = df_film_actor.join(df_film_cat, df_film_actor['film_id'] == df_film_cat['film_id'], 'inner') \
                      .join(df_cat, df_film_cat['category_id'] == df_cat['category_id'], 'inner')
    df = df.where(df['name'] == 'Children').groupBy(df['actor_id'], df['name']).count() \
        .withColumn('num', rank().over(Window.partitionBy(df['name']).orderBy(desc('count'))))
    df = df.where(df['num'] <= 3).join(df_actor, df['actor_id'] == df_actor['actor_id'], 'inner')
    return df.select(df['last_name'], df['first_name'])


# amount of active/inactive customers in each city, sort -> descending inactive
def query_task6(spark: SparkSession):
    df_city = read_table(spark, 'city')
    df_address = read_table(spark, 'address')
    df_customer = read_table(spark, 'customer')

    df = df_address.join(df_customer, df_address['address_id'] == df_customer['address_id'], 'inner')
    active_df = df.where(df['active'] == 1).groupBy(df['city_id']).agg(count(df['customer_id']).alias('active_cnt'))
    inactive_df = df.where(df['active'] == 0).groupBy(df['city_id']).agg(count(df['customer_id']).alias('inactive_cnt'))
    df = df_city.join(active_df, df_city['city_id'] == active_df['city_id'], 'left') \
                .join(inactive_df, df_city['city_id'] == inactive_df['city_id'], 'left')
    df = df.fillna(value=0, subset=['active_cnt', 'inactive_cnt'])

    return df.select(df['city'], df['active_cnt'], df['inactive_cnt']).sort(desc(df['inactive_cnt']))


# show film category with the largest total rental in cities that start with an 'a' or contain '-'
def query_task7(spark: SparkSession):
    df_cat = read_table(spark, 'category')
    df_film_cat = read_table(spark, 'film_category')
    df_film = read_table(spark, 'film')
    df_city = read_table(spark, 'city')
    df_address = read_table(spark, 'address')
    df_inventory = read_table(spark, 'inventory')
    df_customer = read_table(spark, 'customer')
    df_rental = read_table(spark, 'rental')

    df = df_address.join(df_customer, df_customer['address_id'] == df_address['address_id'], 'inner') \
                   .join(df_rental, df_customer['customer_id'] == df_rental['customer_id'], 'inner') \
                   .join(df_inventory, df_rental['inventory_id'] == df_inventory['inventory_id'], 'inner') \
                   .join(df_film, df_inventory['film_id'] == df_film['film_id'], 'inner') \
                   .join(df_film_cat, df_inventory['film_id'] == df_film_cat['film_id'])

    df = df.groupBy(df['city_id'], df['category_id']).agg(sum(df['rental_duration']).alias('total_rental'))
    df = df.join(df_city, df['city_id'] == df_city['city_id'], 'inner') \
           .where((lower(df_city['city']).like('a%')) | (df_city['city'].like('%-%'))) \
           .withColumn('num', row_number().over(Window.partitionBy(df['city_id']).orderBy(desc('total_rental'))))
    df = df.join(df_cat, df['category_id'] == df_cat['category_id'], 'inner').where(df['num'] == 1)

    return df.select(df['city'], df['name'].alias('category_name'))


def main():
    spark = SparkSession \
        .builder \
        .appName("Pidors") \
        .config("spark.jars", "postgresql-42.5.0.jar") \
        .config("spark.executor.extraClassPath", "./postgresql-42.5.0.jar") \
        .getOrCreate()

    query_task1(spark).show(truncate=False)
    query_task2(spark).show(truncate=False)
    query_task3(spark).show(truncate=False)
    query_task4(spark).show(truncate=False)
    query_task5(spark).show(truncate=False)
    query_task6(spark).show(truncate=False)
    query_task7(spark).show(truncate=False)


if __name__ == '__main__':
    main()

